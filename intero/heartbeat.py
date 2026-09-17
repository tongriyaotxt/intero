"""heartbeat.py — M3 心跳：三态变心率 + 意图调度器 + 主动性拍卖 + 反拗期

自主性循环（autonomy loop）：intero 的第二器官。记忆器官回答"记得什么"，
心跳回答"现在要不要说话/做事"——没有心跳的记忆只是档案柜。

四个组件（对话记录的设计共识，本文档为首次工程化）：
  三态变心率：ENGAGED（交互中，快心率）/ WATCH（值守，中心率）/ DREAM（休眠，慢心率），
             心率 = tick 间隔，状态由最近交互时间与显式休眠指令驱动；
  意图调度器：意图带紧迫度与时效（TTL），过期自动作废，永不积灰；
  主动性拍卖：每个 tick 所有意图出价，**沉默也是竞拍者**（有底价）——
             只有出价压过沉默的意图才获准行动，主动性从此有刹车；
  反拗期：   行动后进入冷却窗，窗内所有出价受罚（神经反拗期类比），
             防止复读机式打扰；冷却随未行动 tick 数衰减。

红线：心跳只产"冲动"，不产内容——说什么是 brain/LLM 的事，这里只裁决时机。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class HeartState(Enum):
    ENGAGED = "engaged"   # 交互中：快心率
    WATCH = "watch"       # 值守：中心率
    DREAM = "dream"       # 休眠：慢心率（M5 夜间时钟的窗口）


#: 三态心率（tick 间隔秒；模拟时可压缩）
HEART_RATES = {HeartState.ENGAGED: 1.0, HeartState.WATCH: 10.0, HeartState.DREAM: 60.0}

#: 各状态停留阈值：距上次用户交互超过此秒数即降档
STATE_TRANSITIONS = [
    (60.0, HeartState.ENGAGED),    # 1 分钟内有交互 → ENGAGED
    (600.0, HeartState.WATCH),     # 10 分钟内 → WATCH
    (float("inf"), HeartState.DREAM),
]


@dataclass
class Intention:
    """一条意图（冲动）：想做什么、多急、能放多久。"""

    kind: str                       # 意图类型（remind/followup/dream/...）
    payload: str                    # 内容描述（给 brain 的提示，不是最终话术）
    urgency: float = 0.5            # 基础紧迫度 [0,1]
    ttl: float = 3600.0             # 时效秒数，过期作废
    created_at: float = 0.0         # 0 = 由 Heartbeat.add_intention 盖章

    def expired(self, now: float) -> bool:
        return now - self.created_at > self.ttl

    def bid(self, now: float) -> float:
        """出价 = 紧迫度 × 时效衰减（越临近死线越急，过半衰期后反而衰减）。"""
        age = now - self.created_at
        life = age / max(self.ttl, 1e-9)
        if life < 0.5:
            time_factor = 0.5 + life          # 前半程：0.5 → 1.0 渐强
        else:
            time_factor = 1.5 - life          # 后半程：1.0 → 0.5 衰减
        return max(0.0, self.urgency * time_factor)


@dataclass
class AuctionResult:
    winner: str                     # "silence" 或 intention.kind
    winning_bid: float
    silence_bid: float
    acted: bool                     # 是否真的行动（反拗期可否决）


class Heartbeat:
    """心跳主循环。可注入时钟（测试用手动 tick）。"""

    def __init__(
        self,
        silence_floor: float = 0.55,        # 沉默底价：意图要超过这个才准说话
        refractory_ticks: int = 5,          # 反拗期长度（tick 数）
        refractory_penalty: float = 0.6,    # 反拗期内出价惩罚系数
        now=None,
    ) -> None:
        self.silence_floor = silence_floor
        self.refractory_ticks = refractory_ticks
        self.refractory_penalty = refractory_penalty
        self._now = now or time.time
        self.state = HeartState.WATCH
        self._last_interaction = self._now()
        self._since_action = self.refractory_ticks   # 启动即出反拗期
        self.intentions: list[Intention] = []
        self.log: list[AuctionResult] = []

    # ---- 状态 ----

    def interact(self) -> None:
        """记录一次用户交互（升心率 + 重置状态时钟）。"""
        self._last_interaction = self._now()
        self.state = HeartState.ENGAGED

    def sleep(self) -> None:
        self.state = HeartState.DREAM

    def _refresh_state(self) -> None:
        if self.state == HeartState.DREAM:
            return                      # 休眠只能被 interact() 唤醒
        idle = self._now() - self._last_interaction
        for threshold, state in STATE_TRANSITIONS:
            if idle <= threshold:
                self.state = state
                return

    @property
    def rate(self) -> float:
        return HEART_RATES[self.state]

    # ---- 意图 ----

    def add_intention(self, intention: Intention) -> None:
        if intention.created_at == 0.0:
            intention.created_at = self._now()
        self.intentions.append(intention)

    def _gc(self, now: float) -> None:
        self.intentions = [i for i in self.intentions if not i.expired(now)]

    # ---- 拍卖 ----

    def _in_refractory(self) -> bool:
        return self._since_action < self.refractory_ticks

    def tick(self) -> AuctionResult:
        """跳一下：刷新状态 → 清理过期意图 → 拍卖 → （可能）行动 → 反拗计数。"""
        now = self._now()
        self._refresh_state()
        self._gc(now)

        silence_bid = self.silence_floor
        best: Intention | None = None
        best_bid = 0.0
        for it in self.intentions:
            b = it.bid(now)
            if self._in_refractory():
                b *= 1.0 - self.refractory_penalty
            if b > best_bid:
                best, best_bid = it, b

        acted = best is not None and best_bid > silence_bid
        if acted:
            self.intentions.remove(best)
            self._since_action = 0
            winner = best.kind
        else:
            self._since_action += 1
            winner = "silence"

        result = AuctionResult(
            winner=winner,
            winning_bid=round(best_bid, 4),
            silence_bid=silence_bid,
            acted=acted,
        )
        self.log.append(result)
        return result

    def run(self, ticks: int) -> list[AuctionResult]:
        """连跳 N 下（仿真用；真实部署由事件循环按 self.rate 间隔调用 tick）。"""
        return [self.tick() for _ in range(ticks)]
