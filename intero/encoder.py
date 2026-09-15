"""encoder.py — 可插拔文本编码器。

设计红线：
- 所有 encoder 输出 L2 归一化向量（余弦 = 点积），dim 固定 768；
- 接口只有 encode()，替换 encoder 是改配置不是改代码；
- 内容库永远存原文（见 store.py），换 encoder 可全量重建。

阶梯（见项目计划）：开源 bge → 微调开源 → API → TF-IDF 兜底（零依赖，任何环境可跑）。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

DIM = 768


class Encoder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """返回 (n, dim) L2 归一化 float32。"""


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class TfidfEncoder:
    """字符 n-gram 哈希 TF-IDF。中文无需分词（字 2~4 gram），零依赖兜底。

    弱点：同义改写识别差（探针 margin 大概率不过），仅保证管道可跑。
    """

    name = "tfidf-charng"

    def __init__(self, dim: int = DIM, ngrams=(2, 3, 4)):
        self.dim = dim
        self.ngrams = ngrams
        self.df = np.zeros(dim, dtype=np.float64)
        self.n_docs = 0

    @staticmethod
    def _norm_text(t: str) -> str:
        return re.sub(r"\s+", "", t.lower())

    def _hash(self, g: str) -> int:
        return int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") % self.dim

    def _grams(self, t: str) -> list[str]:
        t = self._norm_text(t)
        out = []
        for n in self.ngrams:
            out.extend(t[i : i + n] for i in range(max(0, len(t) - n + 1)))
        return out or [t or " "]

    def _tf(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for g in self._grams(text):
            v[self._hash(g)] += 1.0
        return np.log1p(v)

    def partial_fit(self, texts: list[str]) -> None:
        for t in texts:
            seen = {self._hash(g) for g in self._grams(t)}
            for h in seen:
                self.df[h] += 1
            self.n_docs += 1

    def _idf(self) -> np.ndarray:
        return np.log((1 + self.n_docs) / (1 + self.df)) + 1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        idf = self._idf()
        return _l2n(np.stack([self._tf(t) * idf for t in texts]))


class STEncoder:
    """sentence-transformers（懒加载，装了才可用）。默认 bge-base-zh-v1.5。"""

    def __init__(self, model: str = "BAAI/bge-base-zh-v1.5"):
        from sentence_transformers import SentenceTransformer  # noqa

        self._m = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = self._m.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class APIEncoder:
    """OpenAI 兼容 embedding API（从 .env / 环境变量读 key）。懒加载 requests。"""

    def __init__(self, model: str, base_url: str, api_key: str, dim: int = DIM):
        self.model, self.base_url, self._key = model, base_url.rstrip("/"), api_key
        self.dim = dim
        self.name = f"api:{model}"

    def encode(self, texts: list[str]) -> np.ndarray:
        import urllib.request, json

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        vecs = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return _l2n(np.asarray(vecs, dtype=np.float32))


# ---------------------------------------------------------------- 探针

#: 中文语义区分探针：过门标准 margin = min(sim(a)) − max(sim(b)) > 0.15
PROBE = {
    "anchor": "我喝咖啡从来不加糖",
    "a_paraphrase": [  # 同义改写：应高相似
        "我喝美式从来不放糖",
        "咖啡我一般都喝无糖的",
        "我的咖啡不加糖，谢谢",
        "喝咖喝啡时我从不放糖进去",
    ],
    "b_same_topic": [  # 同主题异事实：必须显著低于 (a)
        "我喝咖啡要加双份糖",
        "我喝咖啡必须加三包糖才够味",
        "我从来不喝咖啡，只喝茶",
        "我喝咖啡喜欢加很多糖浆",
    ],
    "c_unrelated": [  # 无关：应≈0
        "数据库索引怎么优化",
        "今天下午的会议改到三点",
        "这个函数的返回值类型不对",
    ],
}
PROBE_MARGIN = 0.15


def probe(enc: Encoder) -> dict:
    """跑探针，返回 {margin, sim_a_min, sim_b_max, sim_c_max, passed}。"""
    anchor = enc.encode([PROBE["anchor"]])[0]
    sa = enc.encode(PROBE["a_paraphrase"]) @ anchor
    sb = enc.encode(PROBE["b_same_topic"]) @ anchor
    sc = enc.encode(PROBE["c_unrelated"]) @ anchor
    margin = float(sa.min() - sb.max())
    return {
        "encoder": enc.name,
        "margin": margin,
        "sim_a_min": float(sa.min()),
        "sim_b_max": float(sb.max()),
        "sim_c_max": float(sc.max()),
        "passed": margin > PROBE_MARGIN,
    }


def best_available(prefer_st: bool = True) -> Encoder:
    """阶梯选优：ST(bge) → TF-IDF 兜底。API 需显式配置，不在默认路径。"""
    if prefer_st:
        try:
            return STEncoder()
        except Exception:
            pass
    return TfidfEncoder()
