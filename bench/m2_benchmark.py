"""bench/m2_benchmark.py — M2 三方对拍：sidecar vs RAG vs 全上下文

按 README 预先登记表验收（不许事后改口）：
  | 写入量 vs RAG 全存     | 🟢 预测赢 ~70%（门控构造保证） |
  | 过期内容占比           | 🟢 预测赢（RAG 无遗忘机制）     |
  | flashbulb 集群细节     | 🟡 预测小赢到打平              |
  | 组合事实召回           | 🟡 预测打平（λ 爬升即测量值）   |
  | 孤立事实召回           | 🔴 预测可能小输（门控漏埋点）   |

三臂：
  sidecar：门控写入 + Titans 记忆 + λ 双路读出（生产配置 dim=768 hidden=4096 depth=3）
  rag：    全量写入，纯向量检索（λ=0 路径）
  fullctx：全量保留，检索 = 时间窗内最近 N 字符（模拟有限上下文，超窗旧内容丢弃）

encoder：bge-base-zh（语义真实数字）；归一化前置：剧本句本身是干净模板句，
归一化近似恒等，按 2026-09-16 架构决策三臂共享且此处从略（真实用户话术归一化在 M4 验收）。

运行：HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/Scripts/python.exe bench/m2_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from intero.encoder import STEncoder
from intero.memory import TitansMemory
from intero.store import ContentStore
from bench.scenarios import gen_scenario

FULLCTX_BUDGET_CHARS = 4000   # 全上下文臂的窗口预算
RESULTS = Path("bench/results")


# ---------------------------------------------------------------- 三臂

class Arm:
    name = "arm"

    def ingest(self, text: str, vec: np.ndarray, kind: str) -> None: ...

    def retrieve(self, qvec: np.ndarray, topk: int = 3) -> list[dict]: ...

    def stats(self) -> dict: ...


class SidecarArm(Arm):
    name = "sidecar"

    def __init__(self, store: ContentStore, mem: TitansMemory, dim: int):
        self.st, self.mem = store, mem
        self.dim = dim
        self.n_in = 0
        self.mu = np.zeros(dim, dtype=np.float32)
        self.prewrite: list[float] = []

    def _center(self, raw: np.ndarray) -> np.ndarray:
        self.n_in += 1
        self.mu += (raw - self.mu) / self.n_in
        v = raw - self.mu
        return v / max(np.linalg.norm(v), 1e-12)

    def ingest(self, text, vec, kind):
        v = self._center(vec)
        wrote = self.mem.write(v, v, redundancy=self.st.redundancy(v))
        self.prewrite.append(self.mem.last_prewrite_mse)
        if wrote:
            self.st.add(text, v, kind=kind)
        if self.mem.writes % 32 == 0 and self.prewrite:
            self.st.update_lambda(float(np.mean(self.prewrite[-64:])), 1.0 / self.dim)

    def retrieve(self, qvec, topk=3):
        self.n_in += 1
        self.mu += (qvec - self.mu) / self.n_in
        qv = qvec - self.mu
        qv = qv / max(np.linalg.norm(qv), 1e-12)
        return self.st.retrieve(qv, m_out=self.mem.read(qv), topk=topk)

    def stats(self):
        return {"写入条数": self.mem.writes, "λ": round(self.st.lam, 3),
                "回滚": self.mem.rollbacks, "库存": len(self.st)}


class RagArm(Arm):
    name = "rag"

    def __init__(self, store: ContentStore):
        self.st = store
        self.n_in = 0

    def ingest(self, text, vec, kind):
        self.n_in += 1
        self.st.add(text, vec, kind=kind)

    def retrieve(self, qvec, topk=3):
        return self.st.retrieve(qvec, m_out=None, topk=topk)

    def stats(self):
        return {"写入条数": self.n_in, "库存": len(self.st)}


class FullCtxArm(Arm):
    name = "fullctx"

    def __init__(self, budget: int = FULLCTX_BUDGET_CHARS):
        self.budget = budget
        self.items: list[dict] = []
        self.n_in = 0

    def ingest(self, text, vec, kind):
        self.n_in += 1
        self.items.append({"text": text, "kind": kind})
        while sum(len(i["text"]) for i in self.items) > self.budget and self.items:
            self.items.pop(0)   # 超窗丢最旧（真实上下文行为）

    def retrieve(self, qvec, topk=3):
        # 全上下文"什么都记得窗口内的"，模拟为窗口内全部返回（评分=在窗）
        return [{"text": i["text"], "kind": i["kind"], "score": 1.0} for i in self.items[-topk * 4:]]

    def in_context(self, expect: str) -> bool:
        return any(expect in i["text"] for i in self.items)

    def stats(self):
        return {"写入条数": self.n_in, "窗口字符": sum(len(i["text"]) for i in self.items)}


# ---------------------------------------------------------------- 对拍

def evaluate(arm: Arm, questions: list[dict], enc, topk: int = 3) -> dict:
    """按题型算召回；过期题同时统计 top-k 里过期版本出现率。"""
    per_type: dict[str, list[bool]] = {}
    expired_hits, expired_total = 0, 0
    for q in questions:
        qv = enc.encode([q["q"]])[0]
        if isinstance(arm, FullCtxArm):
            ok = arm.in_context(q["expect"])
            if "expired" in q:
                expired_total += 1
                expired_hits += arm.in_context(q["expired"])
        else:
            res = arm.retrieve(qv, topk=topk)
            ok = any(q["expect"] in r["text"] for r in res)
            if "expired" in q:
                expired_total += 1
                expired_hits += any(q["expired"] in r["text"] for r in res)
        per_type.setdefault(q["qtype"], []).append(bool(ok))
    recall = {t: round(sum(v) / len(v), 3) for t, v in per_type.items()}
    return {
        "recall": recall,
        "expired_ratio": round(expired_hits / max(expired_total, 1), 3),
    }


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(seed)
    sc = gen_scenario(rng)
    stream, questions = sc["stream"], sc["questions"]
    print(f"剧本：{len(stream)} 条流（事实 {sum(1 for s in stream if s['kind'] != 'noise')} + "
          f"噪声 {sum(1 for s in stream if s['kind'] == 'noise')}），{len(questions)} 问\n")

    enc = STEncoder("BAAI/bge-base-zh-v1.5")
    texts = [s["text"] for s in stream] + [q["q"] for q in questions]
    vecs = {t: v for t, v in zip(texts, enc.encode(texts))}

    tmp = tempfile.mkdtemp(prefix="intero_m2_")
    arms: list[Arm] = [
        SidecarArm(ContentStore(os.path.join(tmp, "side.db"), dim=enc.dim),
                   TitansMemory(dim=enc.dim, hidden=4096, depth=3, seed=0), enc.dim),
        RagArm(ContentStore(os.path.join(tmp, "rag.db"), dim=enc.dim)),
        FullCtxArm(),
    ]

    t0 = time.time()
    for s in stream:
        v = vecs[s["text"]]
        for arm in arms:
            arm.ingest(s["text"], v, s["kind"])
    print(f"摄入完成，耗时 {time.time() - t0:.1f}s\n")

    report: dict[str, dict] = {}
    for arm in arms:
        ev = evaluate(arm, questions, enc)
        report[arm.name] = {**ev, **arm.stats()}

    n_in = len(stream)
    rag_writes = report["rag"]["写入条数"]
    side_writes = report["sidecar"]["写入条数"]
    write_saving = 1 - side_writes / rag_writes

    print("== 写入量 ==")
    print(f"  sidecar 写入 {side_writes}/{n_in}（率 {side_writes / n_in:.0%}），"
          f"vs RAG 全存 {rag_writes} → 省 {write_saving:.0%}（预测 🟢 ~70%）")
    print(f"  sidecar λ={report['sidecar']['λ']}  回滚={report['sidecar']['回滚']}")
    print(f"  fullctx 窗口占用 {report['fullctx']['窗口字符']}/{FULLCTX_BUDGET_CHARS} 字符")

    print("\n== 分题型召回@3 ==")
    qtypes = ["isolated", "composite", "flashbulb", "expired"]
    header = f"{'题型':<12}" + "".join(f"{a:>10}" for a in report)
    print(header)
    for t in qtypes:
        row = f"{t:<12}"
        for a in report:
            row += f"{report[a]['recall'].get(t, float('nan')):>10}"
        print(row)

    print("\n== 过期内容占比（top-3 中出现旧版本的比例，越低越好） ==")
    for a in report:
        print(f"  {a:<10}{report[a]['expired_ratio']:>8.3f}")

    print("\n== 对照预先登记表 ==")
    verdicts = {
        "写入量省 ~70%": write_saving >= 0.65,
        "过期占比 sidecar < rag": report["sidecar"]["expired_ratio"] < report["rag"]["expired_ratio"],
        "flashbulb sidecar ≥ rag": report["sidecar"]["recall"].get("flashbulb", 0)
        >= report["rag"]["recall"].get("flashbulb", 0) - 0.05,
        "composite sidecar ≥ rag−0.05": report["sidecar"]["recall"].get("composite", 0)
        >= report["rag"]["recall"].get("composite", 0) - 0.05,
    }
    for k, v in verdicts.items():
        print(f"  {'✅' if v else '❌'} {k}")
    print(f"  （孤立事实预测🔴小输：sidecar {report['sidecar']['recall'].get('isolated')} vs "
          f"rag {report['rag']['recall'].get('isolated')}，实测差 "
          f"{report['sidecar']['recall'].get('isolated', 0) - report['rag']['recall'].get('isolated', 0):+.3f}）")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"m2_report_seed{seed}.json"
    out.write_text(json.dumps({"report": report, "write_saving": write_saving,
                               "verdicts": verdicts}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告已存 {out}")


if __name__ == "__main__":
    main()
