"""bench/probe_oracle.py — 隔离实验：理想归一化（人工书写）下管线能否过门

目的：把"归一化质量"从"管线设计"中隔离。
  - 若人工归一化 → margin > 0.15：管线设计正确，瓶颈只在 LLM 强度 → 换强模型即可；
  - 若仍不过门：管线设计（均值池化/编码方式）本身有病，需改管线而非换 LLM。

运行：PYTHONPATH=. .venv/Scripts/python.exe bench/probe_oracle.py
"""

from __future__ import annotations

import numpy as np

from intero.encoder import PROBE, PROBE_MARGIN, STEncoder

BGE_ZH_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 人工书写的理想原子事实（忠实保留否定与限定词）
ORACLE = {
    "我喝咖啡从来不加糖": ["用户喝咖啡不加糖"],
    "我喝美式从来不放糖": ["用户喝美式咖啡不放糖"],
    "咖啡我一般都喝无糖的": ["用户喝无糖咖啡"],
    "我的咖啡不加糖，谢谢": ["用户的咖啡不加糖"],
    "喝咖喝啡时我从不放糖进去": ["用户喝咖啡不放糖"],
    "我喝咖啡要加双份糖": ["用户喝咖啡加双份糖"],
    "我喝咖啡必须加三包糖才够味": ["用户喝咖啡加三包糖"],
    "我从来不喝咖啡，只喝茶": ["用户不喝咖啡", "用户只喝茶"],
    "我喝咖啡喜欢加很多糖浆": ["用户喝咖啡加很多糖浆"],
    "数据库索引怎么优化": ["数据库索引优化方法"],
    "今天下午的会议改到三点": ["会议改到下午三点"],
    "这个函数的返回值类型不对": ["函数返回值类型错误"],
}


def encode_facts(enc, texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        facts = ORACLE[t]
        v = enc.encode([BGE_ZH_INSTRUCTION + f for f in facts]).mean(axis=0)
        vecs.append(v / max(np.linalg.norm(v), 1e-12))
    return np.stack(vecs).astype(np.float32)


def main() -> None:
    enc = STEncoder("BAAI/bge-base-zh-v1.5")
    sentences = [PROBE["anchor"]] + PROBE["a_paraphrase"] + PROBE["b_same_topic"] + PROBE["c_unrelated"]
    v = encode_facts(enc, sentences)
    anchor, sa, sb, sc = v[0], v[1:5] @ v[0], v[5:9] @ v[0], v[9:12] @ v[0]
    margin = float(sa.min() - sb.max())
    print(f"理想归一化管线: margin={margin:+.4f} "
          f"(a_min={sa.min():.3f} b_max={sb.max():.3f} c_max={sc.max():.3f}) "
          f"→ {'✅ 过门' if margin > PROBE_MARGIN else '❌ 不过门'}")
    print("逐句相似度：")
    for s, sim in zip(PROBE["a_paraphrase"] + PROBE["b_same_topic"], list(sa) + list(sb)):
        print(f"  {sim:.3f}  {s} → {ORACLE[s]}")


if __name__ == "__main__":
    main()
