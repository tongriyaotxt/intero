"""bench/probe_pipeline.py — 归一化+encoder 管线探针（M1.5 改道后的验收口径）

对照三组（同一 bge+指令前缀 encoder）：
  A. 裸 encoder（基线，已知 margin≈+0.001）
  B. 归一化管线：probe 句子先过 LLM 归一化成原子事实，再编码；
     句子的向量 = 其原子事实向量的均值（L2 归一化）
  C. 打印归一化样本人工审阅（归一化质量本身是交付物的一部分）

过门标准不变：margin = min(sim_a) − max(sim_b) > 0.15（验收对象从裸 encoder 变为管线）

运行：HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=. .venv/Scripts/python.exe bench/probe_pipeline.py
"""

from __future__ import annotations

import sys

import numpy as np

from intero.encoder import PROBE, PROBE_MARGIN, STEncoder
from intero.normalize import NullNormalizer, best_available_normalizer

BGE_ZH_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def encode_pipeline(enc, norm, texts: list[str]) -> np.ndarray:
    """原话 → 归一化事实列表 → 逐条编码 → 均值池化 → L2 归一化。"""
    vecs = []
    for t in texts:
        facts = norm.normalize(t) or [t]
        v = enc.encode([BGE_ZH_INSTRUCTION + f for f in facts]).mean(axis=0)
        v = v / max(np.linalg.norm(v), 1e-12)
        vecs.append(v)
    return np.stack(vecs).astype(np.float32)


def probe_vecs(anchor_v, a_v, b_v, c_v) -> dict:
    sa, sb, sc = a_v @ anchor_v, b_v @ anchor_v, c_v @ anchor_v
    margin = float(sa.min() - sb.max())
    return {
        "margin": round(margin, 4),
        "sim_a_min": round(float(sa.min()), 4),
        "sim_b_max": round(float(sb.max()), 4),
        "sim_c_max": round(float(sc.max()), 4),
        "passed": margin > PROBE_MARGIN,
    }


def report(tag: str, r: dict) -> None:
    verdict = "✅ 过门" if r["passed"] else "❌"
    print(f"{verdict} {tag}: margin={r['margin']:+.4f} "
          f"(a_min={r['sim_a_min']:.3f} b_max={r['sim_b_max']:.3f} c_max={r['sim_c_max']:.3f})")


def main() -> None:
    norm = best_available_normalizer() if "--null" not in sys.argv else NullNormalizer()
    enc = STEncoder("BAAI/bge-base-zh-v1.5")

    sentences = [PROBE["anchor"]] + PROBE["a_paraphrase"] + PROBE["b_same_topic"] + PROBE["c_unrelated"]

    print("== A. 裸 encoder（bge-base + 指令前缀，基线） ==")
    raw = enc.encode([BGE_ZH_INSTRUCTION + s for s in sentences])
    report("裸 encoder", probe_vecs(raw[0], raw[1:5], raw[5:9], raw[9:12]))

    print(f"\n== B. 归一化管线（normalizer={norm.name}） ==")
    print("-- 归一化样本（人工审阅） --")
    for s in sentences[:9]:
        print(f"  {s}  →  {norm.normalize(s)}")
    print("-- 管线 margin --")
    pipe = encode_pipeline(enc, norm, sentences)
    report("归一化管线", probe_vecs(pipe[0], pipe[1:5], pipe[5:9], pipe[9:12]))


if __name__ == "__main__":
    main()
