"""drives.py — 内态驱动：没有外部事件时也"想起你"（tech-base/23 的稳态驱动落地 v1）

与意图自生（intents.py）的分工：
  intents.py 从**记忆内容**里抽事件型意图（死线/追问/关心）——有事才想到你；
  drives.py  从**自身状态**里长基线冲动（孤独/好奇/记忆健康）——没事也会想起你。
  这是"像人"的关键差分：人的主动行为大部分是内态表达，不是事件响应。

内态向量 d(t)（每个分量 ∈[0,1]，越高越想说话）：
  social        社交连接 = 距上次用户交互的小时数 / 24（一天没聊 → 满分惦记）
  curiosity     好奇     = 最近 24h 摄入过低（信息饥渴）→ 想主动问点啥
  memory_health 记忆健康 = 矛盾对占比高 → 想找主人仲裁澄清

纪律（防"内态话痨"）：
  - 每个驱动有冷却期（4h），火过一次短时间内不再注册；
  - 注册出的意图走正常拍卖（沉默底价/反拗期/理睬乘子全都生效）——
    内态只负责"起心动念"，说不说仍由心跳裁决；
  - TTL 设短（30min）：内态冲动要么很快熟成开口，要么自然消散——不积灰。
"""

from __future__ import annotations

import json
import time

from .heartbeat import Intention

DRIVE_TTL = 1800.0            # 内态冲动半小时内不熟成就消散
DRIVE_COOLDOWN = 4 * 3600.0   # 同一驱动两次开火的间隔
FIRE_THRESHOLD = 0.6          # 驱动分量超过此值才起心动念

DRIVE_PAYLOADS = {
    "social": "好久没聊了，主动问候一下主人、问问今天在忙什么",
    "curiosity": "最近没什么新信息摄入，主动向主人问一个开放式问题",
    "memory_health": "记忆里有几对矛盾一直没仲裁，找主人确认一下哪个为准",
}


def compute_drives(organ) -> dict[str, float]:
    """从 organ 当前状态算内态向量。"""
    now = time.time()
    idle_h = (now - organ.hb._last_interaction) / 3600.0
    social = min(1.0, idle_h / 24.0)

    day_ago = now - 86400.0
    recent = sum(1 for it in organ.st.items() if it.get("ts", 0) > day_ago)
    curiosity = 1.0 if recent == 0 else (0.5 if recent < 3 else 0.0)

    items = organ.st.items()
    n_conf = sum(1 for it in items if it["kind"] == "conflict")
    health = min(1.0, n_conf / 5.0) if items else 0.0   # 5 对矛盾=满分焦虑

    return {"social": round(social, 3), "curiosity": round(curiosity, 3),
            "memory_health": round(health, 3)}


def maybe_sprout_drive_intentions(organ, now: float | None = None) -> list[str]:
    """内态超阈 → 注册 drive 意图（冷却期去重）。返回本次萌发的驱动名。"""
    now = now or time.time()
    cd = json.loads(organ.st.get_meta_text("drive_cooldown") or "{}")
    fired = []
    for name, value in compute_drives(organ).items():
        if value < FIRE_THRESHOLD:
            continue
        if now - cd.get(name, 0) < DRIVE_COOLDOWN:
            continue
        kind = f"drive:{name}"
        urgency = min(1.0, value * organ.urgency_multiplier(kind))
        organ.hb.add_intention(Intention(
            kind=kind, payload=DRIVE_PAYLOADS[name], urgency=urgency, ttl=DRIVE_TTL))
        cd[name] = now
        fired.append(name)
    if fired:
        organ.st.set_meta_text("drive_cooldown", json.dumps(cd))
        organ.save_heartbeat()
    return fired


def circadian_floor(hour: int) -> float:
    """作息底价：深夜（23-7 点）沉默底价抬到 0.9——人命关天级别才叫醒。"""
    return 0.9 if hour >= 23 or hour < 7 else 0.55
