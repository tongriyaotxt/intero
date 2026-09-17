"""core.py — Intero 门面：把记忆器官粘成一个对象（M4 接入的载体）

管线（写入）：原话 → LLM 归一化（可选）→ encoder → 中心化 → 门控写入 → 内容库
管线（读出）：提问 → encoder → λ 双路检索 → 注入装配（inject.py）

一个 Intero 实例 = 一套记忆器官（memory + store + encoder + normalizer）。
MCP server、prompt 注入、demo 都走这里，不再各自拼零件。
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .encoder import Encoder, best_available
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

    # ---- 向量预处理（与 __main__.py 相同的中心化，破嵌入锥形坍缩） ----

    def _center(self, raw: np.ndarray) -> np.ndarray:
        self.n_in += 1
        self.mu += (raw - self.mu) / self.n_in
        v = raw - self.mu
        return v / max(np.linalg.norm(v), 1e-12)

    # ---- 写入 ----

    def ingest(self, text: str, kind: str = "fact", normalize: bool = True) -> dict:
        """写入一条（可能先归一化）。返回 {facts, written}。"""
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
        """
        from .inject import assemble

        return assemble(self.retrieve(query, topk=topk * 2), topk=topk, budget_chars=budget_chars)

    # ---- 状态 ----

    def status(self) -> dict:
        return {
            "摄入": self.n_in,
            "写入": self.mem.writes,
            "库存": len(self.st),
            "λ": round(self.st.lam, 3),
            "回滚": self.mem.rollbacks,
            "encoder": self.enc.name,
            "normalizer": self.norm.name,
        }
