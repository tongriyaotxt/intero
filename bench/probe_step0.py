"""bench/probe_step0.py — M1.5 Step 0：零训练手段过探针（一次性实验脚本）

候选：
  A. bge-base-zh-v1.5 + query instruction 前缀（全句 / 仅 anchor）
  B. bge-large-zh-v1.5
  C. bge-m3
过门标准与 intero.encoder.probe 一致：margin = min(sim_a) − max(sim_b) > 0.15

运行：HF_ENDPOINT=https://hf-mirror.com .venv/Scripts/python.exe bench/probe_step0.py
"""

from __future__ import annotations

import json
import sys

from intero.encoder import PROBE, PROBE_MARGIN, STEncoder

BGE_ZH_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def probe_with_prefix(enc, prefix_all: str = "", prefix_anchor: str = "") -> dict:
    anchor = enc.encode([prefix_anchor + PROBE["anchor"]])[0]
    sa = enc.encode([prefix_all + t for t in PROBE["a_paraphrase"]]) @ anchor
    sb = enc.encode([prefix_all + t for t in PROBE["b_same_topic"]]) @ anchor
    sc = enc.encode([prefix_all + t for t in PROBE["c_unrelated"]]) @ anchor
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
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "base"):
        enc = STEncoder("BAAI/bge-base-zh-v1.5")
        report("base 无前缀（基线）", probe_with_prefix(enc))
        report("base 全句加指令前缀", probe_with_prefix(enc, prefix_all=BGE_ZH_INSTRUCTION))
        report("base 仅 anchor 加前缀", probe_with_prefix(enc, prefix_anchor=BGE_ZH_INSTRUCTION))
        del enc

    if which in ("all", "large"):
        enc = STEncoder("BAAI/bge-large-zh-v1.5")
        report("large-zh 无前缀", probe_with_prefix(enc))
        report("large-zh 全句加指令前缀", probe_with_prefix(enc, prefix_all=BGE_ZH_INSTRUCTION))
        del enc

    if which in ("all", "m3"):
        enc = STEncoder("BAAI/bge-m3")
        report("m3 无前缀", probe_with_prefix(enc))
        report("m3 全句加指令前缀", probe_with_prefix(enc, prefix_all=BGE_ZH_INSTRUCTION))
        del enc

    if which in ("all", "small"):
        enc = STEncoder("BAAI/bge-small-zh-v1.5")
        report("small-zh 无前缀", probe_with_prefix(enc))
        report("small-zh 全句加指令前缀", probe_with_prefix(enc, prefix_all=BGE_ZH_INSTRUCTION))
        del enc

    if which in ("all", "m3e"):
        enc = STEncoder("moka-ai/m3e-base")
        report("m3e-base 无前缀", probe_with_prefix(enc))
        del enc


if __name__ == "__main__":
    main()
