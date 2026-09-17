"""core.py — Intero 门面：把记忆器官粘成一个对象（M4 接入的载体）

管线（写入）：原话 → LLM 归一化（可选）→ encoder → 中心化 → 门控写入 → 内容库
管线（读出）：提问 → encoder → λ 双路检索 → 注入装配（inject.py）

一个 Intero 实例 = 一套记忆器官（memory + store + encoder + normalizer）。
MCP server、prompt 注入、demo 都走这里，不再各自拼零件。
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from .encoder import Encoder, best_available
from .heartbeat import Heartbeat, Intention
from .intents import ReminderExtractor, best_available_extractor
from .memory import TitansMemory
from .normalize import Normalizer, best_available_normalizer
from .store import ContentStore


class Intero:
    """记忆器官整机。用法：

        intero = Intero()                       # 默认 bge + .env 配置的归一化器
        intero.ingest("我喝咖啡从来不加糖")      # 写入（归一化+门控）
        block = intero.recall("我喝咖啡加不加糖") # → 可直接拼进 prompt 的记忆块
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        normalizer: Normalizer | None = None,
        memory: TitansMemory | None = None,
        store_path: str | None = None,
        lam_max: float = 0.7,
        mem_hidden: int = 4096,
        mem_depth: int = 3,
        heartbeat: Heartbeat | None = None,
        extractor: ReminderExtractor | None = None,
    ) -> None:
        self.enc = encoder or best_available()
        self.norm = normalizer or best_available_normalizer()
        self.mem = memory or TitansMemory(dim=self.enc.dim, hidden=mem_hidden, depth=mem_depth, seed=0)
        # 优先级：显式参数 > 环境变量 INTERO_STORE（MCP/常驻场景持久化）> 临时目录（测试隔离）
        path = (
            store_path
            or os.environ.get("INTERO_STORE")
            or os.path.join(tempfile.mkdtemp(prefix="intero_"), "content.db")
        )
        self.st = ContentStore(path, dim=self.enc.dim, lam_max=lam_max)
        self.n_in = 0
        self.mu = np.zeros(self.enc.dim, dtype=np.float32)
        self._prewrite: list[float] = []
        # 心跳器官：状态随 sqlite 持久化（MCP 宿主 spawn-per-call，进程是短命的）
        self.hb = heartbeat or Heartbeat()
        self.reload_heartbeat()
        self.reminder_extractor = extractor or best_available_extractor()

    # ---- 心跳：每次交互 = 一次交互记录 + 跳一拍（请求驱动宿主的懒惰心跳） ----

    def _beat(self) -> None:
        self.hb.interact()
        self.hb.tick()
        self.save_heartbeat()

    # ---- 意图自生：从库存事实里萌发提醒（dream 周期调用，也可显式调） ----

    def derive_intentions(self, extractor: ReminderExtractor | None = None) -> dict:
        """通读未检查过的库存事实，抽出到期事项注册成心跳意图。可审计可去重。"""
        import time as _time

        ex = extractor or self.reminder_extractor
        seen = set(json.loads(self.st.get_meta_text("derived_fact_ids") or "[]"))
        cands = [it for it in self.st.items()
                 if it["kind"] in ("fact", "consolidated", "composite") and it["id"] not in seen]
        if not cands:
            return {"检查": 0, "萌发": 0, "extractor": ex.name}
        rems = ex.extract([it["text"] for it in cands])
        known = {i.payload for i in self.hb.intentions} | {p.payload for p in self.hb.pending}
        now = _time.time()
        n = 0
        for r in rems:
            payload = r["payload"]
            if payload in known:
                continue
            if r["deadline_ts"] is not None:
                ttl = r["deadline_ts"] - now
                if ttl <= 0:
                    continue                    # 已过期不萌发
            else:
                ttl = 6 * 3600.0                # 无死线：默认半天时效
            self.hb.add_intention(Intention(
                kind="sprout", payload=payload, urgency=r["urgency"], ttl=ttl))
            known.add(payload)
            n += 1
        seen |= {it["id"] for it in cands}      # 无论萌发与否，检查过的不重复付 LLM 费
        self.st.set_meta_text("derived_fact_ids", json.dumps(sorted(seen)))
        self.save_heartbeat()
        return {"检查": len(cands), "萌发": n, "extractor": ex.name}

    # ---- 夜间周期：dream（回放/策展/晋升） + 意图自生 ----

    def dream_cycle(self, verbose: bool = False) -> dict:
        from .dream import dream

        report = dream(self.mem, self.st, verbose=verbose)
        report["意图自生"] = self.derive_intentions()
        return report

    def add_intention(self, kind: str, payload: str, urgency: float = 0.5, ttl: float = 3600.0) -> dict:
        """注册一条意图（冲动）。心跳只裁决时机，说什么由 LLM 决定。"""
        self._beat()
        self.hb.add_intention(Intention(kind=kind, payload=payload, urgency=urgency, ttl=ttl))
        self.save_heartbeat()
        return {"registered": kind, "urgency": urgency, "ttl": ttl, "队列": len(self.hb.intentions)}

    def tick(self) -> dict:
        """显式跳一拍（宿主每轮对话结束可调）。返回拍卖结果。"""
        self._beat()
        r = self.hb.log[-1]
        return {"赢者": r.winner, "行动": r.acted, "状态": self.hb.state.value,
                "待说": len(self.hb.pending), "意图队列": len(self.hb.intentions)}

    # ---- 守护进程接口（daemon.py 用；daemon 不是用户，跳拍不记交互） ----

    def daemon_tick(self) -> dict:
        self.hb.tick()
        self.save_heartbeat()
        r = self.hb.log[-1]
        return {"赢者": r.winner, "行动": r.acted, "状态": self.hb.state.value,
                "待说": len(self.hb.pending), "意图队列": len(self.hb.intentions)}

    def save_heartbeat(self) -> None:
        self.st.set_meta_text("heartbeat", json.dumps(self.hb.snapshot()))

    def reload_heartbeat(self) -> None:
        """从 sqlite 重载心跳快照（多进程共享同一库，每拍前先吸收他进程写入）。"""
        snap = self.st.get_meta_text("heartbeat")
        if snap:
            self.hb.restore(json.loads(snap))

    # ---- 向量预处理（与 __main__.py 相同的中心化，破嵌入锥形坍缩） ----

    def _center(self, raw: np.ndarray) -> np.ndarray:
        self.n_in += 1
        self.mu += (raw - self.mu) / self.n_in
        v = raw - self.mu
        return v / max(np.linalg.norm(v), 1e-12)

    # ---- 写入 ----

    def ingest(self, text: str, kind: str = "fact", normalize: bool = True) -> dict:
        """写入一条（可能先归一化）。返回 {facts, written}。"""
        self._beat()
        facts = self.norm.normalize(text) if normalize else [text]
        if not facts:
            facts = [text]
        written = 0
        for f in facts:
            v = self._center(self.enc.encode([f])[0])
            if self.mem.write(v, v, redundancy=self.st.redundancy(v)):
                self.st.add(f, v, kind=kind)
                written += 1
            self._prewrite.append(self.mem.last_prewrite_mse or 0.0)
            if self.mem.writes % 32 == 0 and self._prewrite:
                self.st.update_lambda(float(np.mean(self._prewrite[-64:])), 1.0 / self.enc.dim)
        return {"facts": facts, "written": written}

    # ---- 读出 ----

    def retrieve(self, query: str, topk: int = 3) -> list[dict]:
        qv = self._center(self.enc.encode([query])[0])
        return self.st.retrieve(qv, m_out=self.mem.read(qv), topk=topk)

    def recall(self, query: str, topk: int = 5, budget_chars: int = 600) -> str:
        """prompt 注入装配：检索 → 记忆块（可直接拼进 system/user prompt）。

        consolidated 优先，dedup/noise 不上墙，conflict 标注供 LLM 裁决。
        待说事项（心跳拍卖赢出的主动冲动）置顶呈现并视为已送达。
        """
        from .inject import assemble

        self._beat()
        block = assemble(self.retrieve(query, topk=topk * 2), topk=topk, budget_chars=budget_chars)
        pending = self.hb.deliver_pending()
        if pending:
            self.save_heartbeat()
            items = "\n".join(f"- [{p.kind}] {p.payload}" for p in pending)
            head = ("【待说事项】以下是你不在场时记忆器官决定要主动提起的事，"
                    "请在回答用户前先自然地处理：\n" + items)
            return head + ("\n\n" + block if block else "")
        return block

    # ---- 状态 ----

    def status(self) -> dict:
        self._beat()
        return {
            "摄入": self.n_in,
            "写入": self.mem.writes,
            "库存": len(self.st),
            "λ": round(self.st.lam, 3),
            "回滚": self.mem.rollbacks,
            "encoder": self.enc.name,
            "normalizer": self.norm.name,
            "心跳": self.hb.state.value,
            "意图": len(self.hb.intentions),
            "待说": len(self.hb.pending),
        }
