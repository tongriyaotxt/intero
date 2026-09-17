"""bench/finetune_encoder.py — 归一化后干净句上的 encoder 轻量微调（扩否定 margin）

背景（见 bench/CALIBRATION.md 改道实录）：归一化管线保证语义忠实与排序，
但 bge 嵌入几何中否定是弱特征（"用户从来不喝咖啡" vs "用户喝咖啡从来不加糖"
余弦高达 0.80+），记忆系统的合并/去重决策边界不安全。
本脚本在归一化风格的干净句上微调 bge-base-zh-v1.5，目标：
矛盾句相对改写的 margin 从 ~0.06 扩到 0.15+。

数据：规则模板量产 (anchor, 正例改写, 硬负例) 三元组——
  硬负例三类：整句否定 / 限定词翻转（不加糖↔加双份糖）/ 主客体替换
  留出 2 个主题族做 test，防过拟合到训练主题（dev/test 双过门才算赢）。

运行：HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/Scripts/python.exe bench/finetune_encoder.py
产物：bench/results/bge-neg-ft/（微调模型）+ 探针前后对比打印
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

OUT_DIR = Path("bench/results/bge-neg-ft")
MODEL = "BAAI/bge-base-zh-v1.5"
INS = "为这个句子生成表示以用于检索相关文章："

# ---------------- 模板量产训练对 ----------------
# 每个主题族：anchor 模板 + 改写模板 + 否定/翻转/替换模板（归一化后的干净句风格）

FAMILIES = [
    {  # 饮品偏好
        "anchor": "用户喝{a}从来不加{b}",
        "pos": ["用户喝{a}不加{b}", "用户的{a}不放{b}", "喝{a}时用户从不加{b}", "用户一般喝无{b}的{a}"],
        "neg_negation": ["用户从来不喝{a}", "用户只喝{c}不喝{a}"],
        "neg_flip": ["用户喝{a}要加双份{b}", "用户喝{a}必须加很多{b}"],
        "neg_swap": ["用户喝{c}从来不加{b}", "用户喝{a}从来不加{d}"],
        "slots": {"a": ["咖啡", "美式", "拿铁", "红茶"], "b": ["糖", "奶", "糖浆"], "c": ["茶", "果汁"], "d": ["盐", "蜂蜜"]},
        "test": False,
    },
    {  # 饮食忌口
        "anchor": "用户吃{a}从来不放{b}",
        "pos": ["用户吃{a}不放{b}", "用户的{a}不加{b}", "吃{a}时用户从不放{b}"],
        "neg_negation": ["用户从来不吃{a}", "用户对{a}过敏"],
        "neg_flip": ["用户吃{a}要放很多{b}", "用户吃{a}必须加{b}"],
        "neg_swap": ["用户吃{c}从来不放{b}", "用户吃{a}从来不放{d}"],
        "slots": {"a": ["面", "火锅", "沙拉"], "b": ["辣", "香菜", "醋"], "c": ["米饭", "面包"], "d": ["糖", "酱油"]},
        "test": False,
    },
    {  # 作息习惯
        "anchor": "用户从来不在{a}工作",
        "pos": ["用户在{a}不工作", "{a}用户从不工作", "用户的{a}不安排工作"],
        "neg_negation": ["用户在{a}工作", "用户只在{a}工作"],
        "neg_flip": ["用户经常在{a}加班", "用户最喜欢{a}工作"],
        "neg_swap": ["用户从来不在{b}工作", "用户在{a}从不运动"],
        "slots": {"a": ["周末", "晚上", "周一早晨"], "b": ["工作日", "下午"]},
        "test": True,  # 留出做 test 族
    },
    {  # 地点事实
        "anchor": "用户住在{a}的{b}附近",
        "pos": ["用户的住处靠近{a}的{b}", "用户家就在{a}{b}旁边", "用户居住在{a}的{b}一带"],
        "neg_negation": ["用户不住在{a}", "用户从没去过{a}"],
        "neg_flip": ["用户住在离{a}很远的地方", "用户刚搬离{a}的{b}"],
        "neg_swap": ["用户住在{c}的{b}附近", "用户住在{a}的{d}附近"],
        "slots": {"a": ["海淀区", "浦东新区", "天河区"], "b": ["地铁站", "公园"], "c": ["朝阳区", "武侯区"], "d": ["医院", "学校"]},
        "test": False,
    },
    {  # 数字/事实
        "anchor": "用户的{a}是{b}",
        "pos": ["{a}为{b}", "用户把{a}定为{b}", "{b}是用户的{a}"],
        "neg_negation": ["用户的{a}不是{b}", "{b}从来不是用户的{a}"],
        "neg_flip": ["用户的{a}是{c}", "用户的{a}刚改成{d}"],
        "neg_swap": ["用户的{e}是{b}", "别人的{a}是{b}"],
        "slots": {"a": ["生日", "工号", "座位号"], "b": ["3月15日", "A1024", "靠窗"], "c": ["5月20日", "B2048"], "d": ["C4096", "过道"], "e": ["纪念日", "房间号"]},
        "test": True,  # 留出做 test 族
    },
    {  # 宠物/人物
        "anchor": "用户的{a}叫{b}",
        "pos": ["用户给{a}起名叫{b}", "{b}是用户{a}的名字", "用户的{a}名叫{b}"],
        "neg_negation": ["用户没有{a}", "用户的{a}不叫{b}"],
        "neg_flip": ["用户的{a}叫{c}", "用户把{a}改名成{c}"],
        "neg_swap": ["用户朋友的{a}叫{b}", "用户的{d}叫{b}"],
        "slots": {"a": ["猫", "狗", "鹦鹉"], "b": ["可乐", "豆豆", "雪球"], "c": ["雪碧", "毛毛"], "d": ["仓鼠", "乌龟"]},
        "test": False,
    },
]


def gen_pairs(seed: int = 0) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """返回 (train_triplets, test_triplets)，每个三元组 (anchor, pos, neg)。"""
    rng = random.Random(seed)
    train, test = [], []
    for fam in FAMILIES:
        slots = fam["slots"]
        combos = []
        # 展开槽位组合（限量防爆）
        keys = list(slots)
        base = fam["anchor"]
        for _ in range(60):
            fill = {k: rng.choice(v) for k, v in slots.items()}
            a = base.format(**{k: fill.get(k, "") for k in keys})
            for tmpl in fam["pos"]:
                p = tmpl.format(**{k: fill.get(k, "") for k in keys})
                for negkind in ("neg_negation", "neg_flip", "neg_swap"):
                    n = rng.choice(fam[negkind]).format(**{k: fill.get(k, "") for k in keys})
                    if n != a and p != a:
                        combos.append((a, p, n))
        rng.shuffle(combos)
        (test if fam["test"] else train).extend(combos[:150])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ---------------- 探针评估 ----------------

def margin_on_triplets(enc, triplets: list[tuple[str, str, str]]) -> dict:
    """三元组版 margin：min(sim(a,p)) − max(sim(a,n))。"""
    if not triplets:
        return {"margin": float("nan")}
    A = enc.encode([INS + a for a, _, _ in triplets])
    P = enc.encode([INS + p for _, p, _ in triplets])
    N = enc.encode([INS + n for _, _, n in triplets])
    sp = np.sum(A * P, axis=1)
    sn = np.sum(A * N, axis=1)
    rank_ok = float((sp > sn).mean())
    return {
        "margin": round(float(sp.min() - sn.max()), 4),
        "margin_mean": round(float(sp.mean() - sn.mean()), 4),
        "rank_acc": round(rank_ok, 4),
    }


def main() -> None:
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    from intero.encoder import PROBE, STEncoder
    from intero.normalize import LLMNormalizer  # noqa: F401  (pipeline 探针可选)

    train, test = gen_pairs()
    print(f"训练三元组 {len(train)}，留出 test {len(test)}")

    # ---- 微调前基线（必须先记录，fit 会原地改模型） ----
    base = SentenceTransformer(MODEL)
    base.max_seq_length = 128
    base_train = margin_on_triplets(base, train[:200])
    base_test = margin_on_triplets(base, test)
    print("[基线] train:", base_train)
    print("[基线] test :", base_test)

    # ---- 微调 ----
    examples = [InputExample(texts=[INS + a, INS + p, INS + n]) for a, p, n in train]
    loader = DataLoader(examples, shuffle=True, batch_size=16, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(base)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.fit(
        train_objectives=[(loader, loss)],
        epochs=2,
        warmup_steps=20,
        optimizer_params={"lr": 2e-5},
        output_path=str(OUT_DIR),
        show_progress_bar=False,
    )

    # ---- 微调后 ----
    ft = SentenceTransformer(str(OUT_DIR))
    ft_train = margin_on_triplets(ft, train[:200])
    ft_test = margin_on_triplets(ft, test)
    print("[微调后] train:", ft_train)
    print("[微调后] test :", ft_test)

    # ---- 回归：无关句相似度应仍低（通用语义没练废） ----
    anchor = ft.encode([INS + PROBE["anchor"]])[0]
    sc = ft.encode([INS + t for t in PROBE["c_unrelated"]]) @ anchor
    print(f"[回归] 无关句 max sim = {float(sc.max()):.3f}（应 < 0.5）")

    summary = {
        "model": str(OUT_DIR),
        "base_test": base_test,
        "ft_test": ft_test,
        "unrelated_max_sim": round(float(sc.max()), 4),
    }
    (OUT_DIR / "finetune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
