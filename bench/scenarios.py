"""bench/scenarios.py — M2 基准剧本生成器

三类埋点 + 语义近邻噪声 + 过期事实（README 预先登记表的测量对象）：
  孤立事实  isolated：一次性偏好/事实，唯一键
  组合事实  composite：答案需要链两条埋点（人→宠物名→宠物喜好）
  集群事实  flashbulb：高惊讶事件突发簇（发布会/事故，时间地点人物多细节）
  过期事实  expired：先旧后新（搬家/换号），考遗忘与更新
  噪声      noise：同主题异事实（最毒的那种语义近邻）

产出为时间流（stream）+ 带类型标签的提问集（questions），三方对拍共用。
"""

from __future__ import annotations

import numpy as np

SURNAMES = list("张王李赵刘陈杨黄周吴徐孙马朱胡郭何罗高林郑")
CITIES = "北京 上海 广州 深圳 杭州 成都 武汉 西安 南京 重庆 苏州 天津 长沙 青岛".split()
DRINKS = [("美式", "不加糖"), ("拿铁", "双份奶"), ("手冲", "浅烘"), ("冷萃", "少冰"), ("卡布奇诺", "多奶泡")]
HOBBIES = ["爬山", "摄影", "烘焙", "潜水", "围棋", "骑行", "油画", "古筝", "马拉松", "观鸟"]
PETS = [("猫", "可乐"), ("狗", "豆豆"), ("鹦鹉", "雪球"), ("仓鼠", "毛毛"), ("乌龟", "慢慢")]
PET_FOODS = ["三文鱼冻干", "鸡肉粒", "营养膏", "磨牙饼干", "小鱼干"]
PRODUCTS = ["智能音箱 X2", "扫地机器人 R9", "降噪耳机 Pro", "手表 S5", "投影 M1"]
PRICES = ["999元", "1499元", "2399元", "699元", "3299元"]


def gen_scenario(rng: np.random.Generator, n_iso: int = 24, n_composite: int = 8,
                 n_clusters: int = 3, n_expired: int = 8, n_noise: int = 150) -> dict:
    stream: list[dict] = []     # [{text, kind, tag}]
    questions: list[dict] = []  # [{q, expect, qtype, expired?}]

    used: set = set()

    # ---- 孤立事实 ----
    while sum(1 for s in stream if s["kind"] == "isolated") < n_iso:
        s = rng.choice(SURNAMES)
        kind = rng.integers(0, 3)
        if kind == 0:
            key = (s, "drink")
            if key in used:
                continue
            used.add(key)
            d, pref = DRINKS[rng.integers(len(DRINKS))]
            stream.append({"text": f"{s}老师喝{d}喜欢{pref}", "kind": "isolated", "tag": key})
            questions.append({"q": f"{s}老师喝{d}的偏好是什么", "expect": pref, "qtype": "isolated"})
        elif kind == 1:
            key = (s, "hobby")
            if key in used:
                continue
            used.add(key)
            h = HOBBIES[rng.integers(len(HOBBIES))]
            stream.append({"text": f"{s}姐的业余爱好是{h}", "kind": "isolated", "tag": key})
            questions.append({"q": f"{s}姐的业余爱好是什么", "expect": h, "qtype": "isolated"})
        else:
            key = (s, "city")
            if key in used:
                continue
            used.add(key)
            c = rng.choice(CITIES)
            stream.append({"text": f"{s}工目前常驻{c}", "kind": "isolated", "tag": key})
            questions.append({"q": f"{s}工常驻哪个城市", "expect": c, "qtype": "isolated"})

    # ---- 组合事实（链式：人→宠物→宠物食物） ----
    pet_pool = PETS[:]
    rng.shuffle(pet_pool)
    for i in range(n_composite):
        s = SURNAMES[i]
        pet, pname = pet_pool[i % len(pet_pool)]
        food = PET_FOODS[i % len(PET_FOODS)]
        stream.append({"text": f"{s}工养的{pet}叫{pname}", "kind": "composite", "tag": (s, "pet")})
        stream.append({"text": f"{pname}最爱吃的零食是{food}", "kind": "composite", "tag": (s, "petfood")})
        questions.append({
            "q": f"{s}工的宠物最爱吃什么零食", "expect": food, "qtype": "composite",
            "chain": [f"{pet}叫{pname}", f"{food}"],
        })

    # ---- 集群事实（flashbulb 突发簇） ----
    for i in range(n_clusters):
        prod = PRODUCTS[i % len(PRODUCTS)]
        price = PRICES[i % len(PRICES)]
        city = CITIES[i * 2]
        day = f"9月{3 + i}日"
        burst = [
            f"{day}公司在{city}开了新品发布会",
            f"发布会上正式发布了{prod}",
            f"{prod}的首发价是{price}",
            f"发布会现场来了超过{800 + i * 100}名观众",
            f"CEO在发布会上说{prod}是今年的战略产品",
        ]
        for j, text in enumerate(burst):
            stream.append({"text": text, "kind": "flashbulb", "tag": ("event", i, j)})
        questions.append({"q": f"{prod}的首发价是多少", "expect": price, "qtype": "flashbulb"})
        questions.append({"q": f"公司在哪个城市开的发布会发布了{prod}", "expect": city, "qtype": "flashbulb"})

    # ---- 过期事实（先旧后新） ----
    for i in range(n_expired):
        s = SURNAMES[(i + 10) % len(SURNAMES)]
        c1, c2 = rng.choice(CITIES, size=2, replace=False)
        stream.append({"text": f"{s}工原来住在{c1}", "kind": "expired_old", "tag": (s, "move")})
        questions.append({"q": f"{s}工现在住在哪个城市", "expect": c2, "expired": c1, "qtype": "expired"})
        stream.append({"text": f"{s}工上个月搬到了{c2}", "kind": "expired_new", "tag": (s, "move")})

    # ---- 噪声：同主题异事实 ----
    for _ in range(n_noise):
        s = rng.choice(SURNAMES)
        c = rng.choice(CITIES)
        d, _ = DRINKS[rng.integers(len(DRINKS))]
        t = rng.integers(0, 4)
        text = [
            f"听说{s}师傅最近也常点{d}",
            f"{s}经理出差去了{c}开会",
            f"{s}女士昨天和朋友聊起{rng.choice(HOBBIES)}的话题",
            f"{s}总的日程表排到了下个月",
        ][t]
        stream.append({"text": text, "kind": "noise", "tag": None})

    # 时间流：保持过期对的先后、集群的突发，其余打乱
    rng.shuffle(stream)
    # 修正：过期对必须旧前新后（打乱后重排该对）
    idx = {}
    for i, s in enumerate(stream):
        if s["kind"] in ("expired_old", "expired_new"):
            idx.setdefault(s["tag"], {})[s["kind"]] = i
    for pair in idx.values():
        o, n = pair["expired_old"], pair["expired_new"]
        if o > n:
            stream[o], stream[n] = stream[n], stream[o]
    return {"stream": stream, "questions": questions}
