"""longitudinal.py — 纵向模拟：记忆系统在"生活规模"下活两周（M2 的规模化续集）

动机（2026-09-17 架构讨论的致命级缺口）：
  M2 是玩具规模（每题型 n=4~8，库存几十条）。记忆系统真正的敌人是
  时间和堆积——库存上千后检索质量、矛盾堆积、策展行为全是未知数。

设计：
  语料：DeepSeek 一次性批量生成（缓存 bench/results/longi_corpus.json）——
        planted: 用户事实+改写提问对（5 类：偏好/计划/社交/工作/琐事）
        noise:   与用户无关的世界事实（信息流噪声）
  模拟：14 天，每天 ingest 一批 planted+noise，夜间 dream_cycle（不跑意图自生，
        省 API；意图自生单独测过）。语料事实已是原子事实，ingest 用 NullNormalizer
        （归一化质量已由探针单独验收，此处不重复付费）。
  双臂：sidecar（门控）vs rag（全写入，λ=0 纯向量检索）——M2 对拍的规模化。
  检查点：每 2 天测一次——对**已写入**的 planted 提问，hit@5；
        库存/kind 分布/写入率/λ/矛盾对。

红线： planted 的 query 是"用户后来可能怎么问"的改写（生成时与事实配对），
        命中判定 = 目标事实文本出现在 top5（严格子串）。

用法：
  python bench/longitudinal.py --generate   # 生成语料（~6 次 API 调用）
  python bench/longitudinal.py --run        # 跑模拟（CPU，~10-30 分钟）
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intero.core import Intero
from intero.encoder import STEncoder
from intero.intents import NullReminderExtractor
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer, _load_dotenv

CORPUS_PATH = Path("bench/results/longi_corpus.json")
REPORT_PATH = Path("bench/results/longi_report.json")
DAYS = 14
PLANTED_PER_DAY = 25
NOISE_PER_DAY = 20
CHECKPOINT_EVERY = 2

CATEGORIES = {
    "偏好": "用户的长期偏好（饮食/音乐/运动/作息/审美，如：用户喝咖啡从来不加糖）",
    "计划": "用户的具体计划/日程（含时间，如：用户下周二下午3点前要交知乎初稿）",
    "社交": "用户的社交关系事实（家人/朋友/同事及其细节，如：用户的室友小林对猫毛过敏）",
    "工作": "用户的工作相关事实（项目/工具/进展/烦恼，如：用户负责的服务化改造本周要上线）",
    "琐事": "用户的日常琐事（一次性小事，如：用户昨天把伞落在公司了）",
}


# ---------------- 语料生成（一次性，缓存） ----------------

def _call_api(prompt: str, cfg: dict) -> str:
    req = urllib.request.Request(
        f"{cfg['INTERO_LLM_BASE_URL'].rstrip('/')}/chat/completions",
        data=json.dumps({
            "model": cfg.get("INTERO_LLM_MODEL", "deepseek-chat"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
        }).encode(),
        headers={
            "Authorization": f"Bearer {cfg['INTERO_LLM_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _extract_json_array(text: str) -> list:
    import re
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def generate() -> None:
    cfg = _load_dotenv()
    assert cfg.get("INTERO_LLM_API_KEY"), ".env 缺 INTERO_LLM_API_KEY"
    n_per_cat = DAYS * PLANTED_PER_DAY // len(CATEGORIES)   # 14*25/5 = 70/类
    planted: list[dict] = []
    if CORPUS_PATH.exists():                               # 断点续生成：只补缺
        old = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        planted = old.get("planted", [])
    for cat, desc in CATEGORIES.items():
        have = sum(1 for p in planted if p["category"] == cat)
        if have >= n_per_cat:
            print(f"[gen] {cat} 已有 {have} 条，跳过", flush=True)
            continue
        print(f"[gen] {cat} ×{n_per_cat} ...", flush=True)
        prompt = f"""生成 {n_per_cat} 条关于同一个虚拟用户的事实及对应提问。类别：{desc}
要求：
1. 每条输出 {{"fact": "以'用户'开头的原子事实句", "query": "用户日后可能怎么问起这件事（自然问句，不照搬原文用词）"}}
2. 事实之间彼此独立、细节各异（不同对象/数值/场景），禁止换汤不换药；
3. 只输出 JSON 数组，不要解释。"""
        arr = _extract_json_array(_call_api(prompt, cfg))
        for it in arr:
            if isinstance(it, dict) and it.get("fact") and it.get("query"):
                planted.append({"category": cat, "fact": it["fact"].strip(),
                                "query": it["query"].strip()})
        print(f"[gen]   得到 {sum(1 for p in planted if p['category'] == cat)} 条", flush=True)

    # 噪声：分批生成（单批太长会被截断导致 JSON 解析失败）
    noise: list[str] = []
    if CORPUS_PATH.exists():
        noise = json.loads(CORPUS_PATH.read_text(encoding="utf-8")).get("noise", [])
    batch = 70
    while len(noise) < DAYS * NOISE_PER_DAY:
        print(f"[gen] 噪声 {len(noise)}/{DAYS * NOISE_PER_DAY} ...", flush=True)
        prompt = f"""生成 {batch} 条与用户无关的一般性事实/资讯句（科技、历史、生活常识、新闻感）。
要求：每条一句话、彼此独立、主题分散；只输出 JSON 字符串数组，不要解释。"""
        got = [s.strip() for s in _extract_json_array(_call_api(prompt, cfg))
               if isinstance(s, str) and s.strip()]
        if not got:
            print("[gen] 噪声批次解析失败，重试一次", flush=True)
            got = [s.strip() for s in _extract_json_array(_call_api(prompt, cfg))
                   if isinstance(s, str) and s.strip()]
        noise.extend(got)
        if not got:
            print("[gen] 噪声连续失败，放弃剩余", flush=True)
            break

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(
        json.dumps({"planted": planted, "noise": noise}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[gen] 语料落盘 {CORPUS_PATH}: planted={len(planted)} noise={len(noise)}")


# ---------------- 双臂 ----------------

class SidecarArm:
    """full（默认，Titans 门控）或 light（INTERO_MODE=light，显著∨新颖门控）。"""

    def __init__(self, tmp: str):
        enc = STEncoder()   # bge-base，生产同款
        self.light = os.environ.get("INTERO_MODE") == "light"
        self.name = "sidecar-light" if self.light else "sidecar"
        self.organ = Intero(
            encoder=enc, normalizer=NullNormalizer(),
            memory=None if self.light else TitansMemory(dim=enc.dim, hidden=4096, depth=3, seed=0),
            store_path=os.path.join(tmp, "sidecar.db"),
            extractor=NullReminderExtractor())
        self.writes = 0

    def ingest(self, text: str) -> None:
        r = self.organ.ingest(text, normalize=False)
        self.writes += r["written"]

    def night(self) -> dict:
        from intero.dream import dream
        return dream(self.organ.mem, self.organ.st)

    def query(self, q: str, topk: int = 5) -> list[str]:
        return [r["text"] for r in self.organ.retrieve(q, topk=topk)]

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for it in self.organ.st.items():
            kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
        return {"库存": len(self.organ.st),
                "写入": self.organ.mem.writes if self.organ.mem else "-",
                "λ": round(self.organ.st.lam, 3), "kinds": kinds}


class RagArm:
    """全写入 + 纯向量检索（λ=0，无 Titans 读出、无门控、无 dream）。"""

    name = "rag"

    def __init__(self, tmp: str):
        from intero.store import ContentStore
        self.enc = STEncoder()
        self.st = ContentStore(os.path.join(tmp, "rag.db"), dim=self.enc.dim, lam_max=0.0)
        self.n = 0
        self.mu = None

    def _center(self, raw):
        import numpy as np
        self.n += 1
        self.mu = raw if self.mu is None else self.mu + (raw - self.mu) / self.n
        v = raw - self.mu
        return v / max(float(__import__("numpy").linalg.norm(v)), 1e-12)

    def ingest(self, text: str) -> None:
        self.st.add(text, self._center(self.enc.encode([text])[0]))

    def night(self) -> dict:
        return {}

    def query(self, q: str, topk: int = 5) -> list[str]:
        return [r["text"] for r in self.st.retrieve(self._center(self.enc.encode([q])[0]),
                                                    m_out=None, topk=topk)]

    def stats(self) -> dict:
        return {"库存": len(self.st)}


# ---------------- 模拟主循环 ----------------

def run() -> None:
    import tempfile
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    planted, noise = corpus["planted"], corpus["noise"]
    rng = random.Random(42)
    rng.shuffle(planted)
    rng.shuffle(noise)

    tmp = tempfile.mkdtemp(prefix="intero_longi_")
    arms = [SidecarArm(tmp), RagArm(tmp)]
    asked: list[dict] = []          # 已 ingested 的 planted（含 query）
    report = {"days": DAYS, "checkpoints": []}
    p_i = n_i = 0

    for day in range(1, DAYS + 1):
        t0 = time.time()
        batch_p = planted[p_i:p_i + PLANTED_PER_DAY]
        batch_n = noise[n_i:n_i + NOISE_PER_DAY]
        p_i += len(batch_p)
        n_i += len(batch_n)
        stream = [p["fact"] for p in batch_p] + batch_n
        rng.shuffle(stream)
        for arm in arms:
            for text in stream:
                arm.ingest(text)
        asked.extend(batch_p)
        for arm in arms:
            arm.night()
        print(f"[day {day:02d}] ingest {len(stream)} 条 ({time.time() - t0:.0f}s)", flush=True)

        if day % CHECKPOINT_EVERY == 0 or day == DAYS:
            for arm in arms:
                hits = 0
                for p in asked:
                    if any(p["fact"] in t for t in arm.query(p["query"])):
                        hits += 1
                ck = {"day": day, "arm": arm.name, "提问数": len(asked),
                      "hit@5": round(hits / max(1, len(asked)), 3), **arm.stats()}
                report["checkpoints"].append(ck)
                print(f"  [{arm.name}] hit@5={ck['hit@5']} 库存={ck['库存']} {ck.get('kinds', '')}",
                      flush=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[done] 报告落盘 {REPORT_PATH}")


if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate()
    elif "--run" in sys.argv:
        run()
    else:
        print(__doc__)
