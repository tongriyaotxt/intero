"""冒烟 demo：python -m intero

竖片验收场景：100 条事实 + 300 条语义近邻噪声灌入 → 10 个提问。
报告：写入量（门控应省 ~70%）、Hit@3、λ 爬升、自重构误差、‖S‖（flashbulb 读数）。

注意：默认 TF-IDF encoder 只验证管道（词汇重叠可检索），
真实中文语义区分需 ST/API encoder（先跑 python -m intero.encoder 看探针结果）。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from .encoder import TfidfEncoder
from .memory import TitansMemory
from .store import ContentStore

SURNAMES = list("张王李赵刘陈杨黄周吴徐孙马朱胡郭何罗高林郑")
CITIES = list("北京 上海 广州 深圳 杭州 成都 武汉 西安 南京 重庆 苏州 天津 长沙 青岛 宁波 厦门".split())
DRINKS = [("美式", "不加糖"), ("拿铁", "双份奶"), ("手冲", "浅烘"), ("冷萃", "少冰"), ("卡布奇诺", "多奶泡")]
HOBBIES = ["爬山", "摄影", "烘焙", "潜水", "围棋", "骑行", "油画", "古筝", "马拉松", "观鸟"]


def gen_corpus(rng: np.random.Generator, n_facts=100, n_noise=300, n_q=10):
    facts, qpool = [], []
    used = set()
    q_used = set()  # 提问键去重：保证每个问题在语料中只有唯一答案

    def add_q(s, kind, q, a):
        if (s, kind) not in q_used:
            q_used.add((s, kind))
            qpool.append((q, a))
    while len(facts) < n_facts:
        s = rng.choice(SURNAMES)
        kind = rng.integers(0, 3)
        if kind == 0:
            d, pref = DRINKS[rng.integers(len(DRINKS))]
            key = (s, "drink")
            if key in used:
                continue
            used.add(key)
            facts.append(f"{s}老师喝{d}喜欢{pref}")
            add_q(s, "drink", f"{s}老师喝{d}的偏好是什么", pref)
        elif kind == 1:
            c1, c2 = rng.choice(CITIES, size=2, replace=False)
            key = (s, "move")
            if key in used:
                continue
            used.add(key)
            facts.append(f"{s}工从{c1}搬家到了{c2}")
            add_q(s, "move", f"{s}工现在住在哪个城市", c2)
        else:
            h = HOBBIES[rng.integers(len(HOBBIES))]
            key = (s, "hobby", h)
            if key in used:
                continue
            used.add(key)
            facts.append(f"{s}姐的业余爱好是{h}")
            add_q(s, "hobby", f"{s}姐的业余爱好是什么", h)
    # 噪声：同主题异事实（语义近邻，最毒的那种）
    noise = []
    while len(noise) < n_noise:
        s = rng.choice(SURNAMES)
        c = rng.choice(CITIES)
        d, pref = DRINKS[rng.integers(len(DRINKS))]
        t = rng.integers(0, 3)
        noise.append([
            f"听说{s}师傅最近也常点{d}",
            f"{s}经理出差去了{c}开会",
            f"{s}女士昨天和朋友聊起{rng.choice(HOBBIES)}的话题",
        ][t])
    picked = rng.choice(len(qpool), size=min(n_q, len(qpool)), replace=False)
    questions = [qpool[i] for i in picked]
    return facts, noise, questions


def main():
    rng = np.random.default_rng(0)
    facts, noise, questions = gen_corpus(rng)
    print(f"语料：{len(facts)} 事实 + {len(noise)} 噪声 + {len(questions)} 提问\n")

    enc = TfidfEncoder()
    enc.partial_fit(facts + noise + [q for q, _ in questions])

    tmp = tempfile.mkdtemp(prefix="intero_demo_")
    mem = TitansMemory(dim=enc.dim, hidden=4096, depth=3, seed=0)   # 生产配置 ~25M
    st = ContentStore(os.path.join(tmp, "content.db"), dim=enc.dim)

    stream = facts + noise
    rng.shuffle(stream)
    t0 = time.time()
    n_candidates = 0
    mu = np.zeros(enc.dim, dtype=np.float32)   # 运行均值：中心化破嵌入锥形坍缩
    prewrite: list[float] = []
    chance = 1.0 / enc.dim
    for text in stream:
        raw = enc.encode([text])[0]
        n_candidates += 1
        mu += (raw - mu) / n_candidates
        vec = raw - mu
        vec = vec / max(np.linalg.norm(vec), 1e-12)
        wrote = mem.write(vec, vec, redundancy=st.redundancy(vec))
        prewrite.append(mem.last_prewrite_mse)
        if wrote:
            st.add(text, vec, kind="fact" if text in facts else "noise")
        if mem.writes % 32 == 0 and prewrite:
            st.update_lambda(float(np.mean(prewrite[-64:])), chance)
    st.update_lambda(float(np.mean(prewrite[-64:])), chance)

    dt = time.time() - t0
    print(f"摄入 {n_candidates} 条 → 实际写入 {mem.writes} 条"
          f"（写入率 {mem.writes/n_candidates:.0%}，门控目标 ≤30%）  耗时 {dt:.1f}s")
    print(f"λ 信任权重爬升至 {st.lam:.2f}（0=纯RAG，越高=参数化路由越可信）")
    print(f"自重构误差 {mem.recon_error():.4f}   ‖S‖(flashbulb读数) {mem.past_surprise_norm:.3f}"
          f"   回滚次数 {mem.rollbacks}\n")

    hits = hits_rag = 0
    for q, expect in questions:
        qraw = enc.encode([q])[0]
        qv = qraw - mu
        qv = qv / max(np.linalg.norm(qv), 1e-12)
        res = st.retrieve(qv, m_out=mem.read(qv), topk=3)
        res_rag = st.retrieve(qv, m_out=None, topk=3)   # λ=0 对照组（消融内置）
        ok = any(expect in r["text"] for r in res)
        ok_rag = any(expect in r["text"] for r in res_rag)
        hits += ok
        hits_rag += ok_rag
        mark = "[OK]" if ok else "[X]"
        top = res[0]["text"][:24] if res else "(空)"
        score = f"{res[0]['score']:.3f}" if res else "-"
        print(f"{mark} Q: {q}\n   期望[{expect}] top1: {top} ({score})")
    print(f"\nHit@3：λ混合={hits}/10   纯RAG对照={hits_rag}/10")
    print(f"\n（说明：TF-IDF 下本测试验证的是管道与门控；encoder 语义能力见探针）")


if __name__ == "__main__":
    main()
