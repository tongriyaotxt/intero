"""store.py — 内容库：原文 + 嵌入 + 双路读出（λ 信任混合）。

红线：永远存原文，不只存向量——换 encoder 时可全量重建（迁移路径）。
冷启动双路读出（对论文读出路径的工程化，偏离#3）：
    score_i = λ·cos(M(q), v_i) + (1−λ)·cos(q, v_i)
λ 初始 0（=朴素 RAG，天然对照组），随自重构误差下降而爬升——
"参数化泛化体现在路由"从诚实标注变成可测量项。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np


class ContentStore:
    def __init__(self, path: str, dim: int = 768, lam_max: float = 0.7):
        self.path = path
        self.dim = dim
        self.lam_max = lam_max
        self.lam = 0.0                 # 信任权重：冷启动 = 0（纯 RAG）
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS items("
            "id INTEGER PRIMARY KEY, text TEXT, kind TEXT, ts REAL, deleted INTEGER DEFAULT 0)"
        )
        self._vecs: list[np.ndarray] = []   # 与 id 顺序对齐（含已删位，查询时屏蔽）
        self._ids: list[int] = []
        self._vec_path = path + ".vecs.json"
        self._load_vecs()

    # ---- 写入 ----

    def add(self, text: str, vec: np.ndarray, kind: str = "fact") -> int:
        cur = self._db.execute(
            "INSERT INTO items(text, kind, ts) VALUES (?,?,?)", (text, kind, time.time())
        )
        i = cur.lastrowid
        self._db.commit()
        self._ids.append(i)
        self._vecs.append(np.asarray(vec, dtype=np.float32))
        return i

    def items(self) -> list[dict]:
        """存活条目（含向量），M5 策展/晋升用。"""
        live = self._live_ids()
        return [
            {"id": i, "text": r[0], "kind": r[1], "vec": self._vecs[self._ids.index(i)]}
            for i in self._ids
            if i in live and (r := self._db.execute(
                "SELECT text, kind FROM items WHERE id=?", (i,)).fetchone())
        ]

    def update_kind(self, item_id: int, kind: str) -> None:
        self._db.execute("UPDATE items SET kind=? WHERE id=?", (kind, item_id))
        self._db.commit()

    def delete(self, item_id: int) -> None:
        """删除 = 原文抹除 + 向量屏蔽（crypto-shredding 的轻量版：数据本体不可读）。"""
        self._db.execute("UPDATE items SET text='', deleted=1 WHERE id=?", (item_id,))
        self._db.commit()
        if item_id in self._ids:
            self._vecs[self._ids.index(item_id)] = np.zeros(self.dim, dtype=np.float32)

    def redundancy(self, vec: np.ndarray, window: int = 32) -> float:
        """与最近 window 条的最大余弦（喂给 α 门的冗余特征）。"""
        if not self._vecs:
            return 0.0
        recent = np.stack(self._vecs[-window:])
        return float((recent @ vec).max())

    # ---- 双路读出 ----

    def retrieve(self, q_vec: np.ndarray, m_out: np.ndarray | None = None, topk: int = 3) -> list[dict]:
        if not self._vecs:
            return []
        V = np.stack(self._vecs)
        score = V @ q_vec
        if m_out is not None and self.lam > 0:
            m = m_out / max(np.linalg.norm(m_out), 1e-12)
            score = self.lam * (V @ m) + (1 - self.lam) * score
        rows = self._db.execute(
            f"SELECT id, text, kind FROM items WHERE deleted=0 AND id IN "
            f"({','.join(map(str, self._ids)) or '0'})"
        ).fetchall()
        meta = {r[0]: (r[1], r[2]) for r in rows}
        order = np.argsort(-score)
        out = []
        for idx in order:
            i = self._ids[idx]
            if i not in meta:
                continue
            out.append({"id": i, "text": meta[i][0], "kind": meta[i][1], "score": float(score[idx])})
            if len(out) >= topk:
                break
        return out

    # ---- λ 信任权重：由泛化误差驱动 ----

    def update_lambda(self, prewrite_mse: float | None, chance_mse: float) -> float:
        """λ = 1 − 泛化误差/随机水平。误差≈随机 → λ≈0（自动退化为纯 RAG）；误差→0 → λ→上限。

        单位归一化 v 的随机水平 chance_mse = 1/dim（输出≈0 时的期望每维均方误差）。
        驱动量必须用**写入前**损失（对未见过样本的预测），不是写入后重构——
        否则背下训练集就把 λ 骗到上限（已发生的教训）。
        """
        if prewrite_mse is None or chance_mse <= 0:
            return self.lam
        self.lam = float(np.clip(1.0 - prewrite_mse / chance_mse, 0.0, self.lam_max))
        return self.lam

    # ---- 持久化 ----

    def save(self) -> None:
        with open(self._vec_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self._ids, "lam": self.lam,
                       "vecs": [v.tolist() for v in self._vecs]}, f)

    def _load_vecs(self) -> None:
        if not os.path.exists(self._vec_path):
            return
        with open(self._vec_path, encoding="utf-8") as f:
            d = json.load(f)
        self._ids = d["ids"]
        self._vecs = [np.asarray(v, dtype=np.float32) for v in d["vecs"]]
        self.lam = d.get("lam", 0.0)

    def __len__(self) -> int:
        return sum(1 for i in self._ids if i in self._live_ids())

    def close(self) -> None:
        self._db.close()

    def _live_ids(self) -> set[int]:
        rows = self._db.execute("SELECT id FROM items WHERE deleted=0").fetchall()
        return {r[0] for r in rows}
