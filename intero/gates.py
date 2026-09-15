"""gates.py — Titans 手工门控（对论文学习门的工程替代，偏离#1，诚实标注）。

每个门只吃一个有明确语义的特征，不搞黑盒调参：
  θ_t（学习率）  ← 归一化瞬时惊讶 z_s        越惊讶学得越狠
  η_t（动量保留）← 滑动窗惊讶密度 d_t        连续惊讶→整段牢记（flashbulb）
  α_t（遗忘）    ← 与近 k 条嵌入最大余弦 r_t  冗余内容加速遗忘腾容量
写入门控：z_s 超 running 分位数 τ 才触发更新（省写入的构造保证）。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateParams:
    # —— 以下为标定台冠军值（bench/CALIBRATION.md），冻结，谁改谁举证 ——
    theta_base: float = 0.1     # 学习率下限
    theta_max: float = 0.3      # 学习率上限
    eta_min: float = 0.6        # 动量保留下限
    eta_max: float = 0.9        # 动量保留上限（flashbulb 时）
    alpha_base: float = 0.001   # 基础遗忘
    alpha_r: float = 0.02       # 冗余加成遗忘
    tau_quantile: float = 0.70  # 写入门分位数（≈省 70% 写入的旋钮）
    window: int = 64            # 惊讶统计窗
    grad_clip: float = 1.0      # 全局梯度范数上限（保险）
    theta_clip: float = 0.6     # 单步学习率硬上限（保险）


class RunningStats:
    """Welford 在线均值/方差 + 滑动窗分位数。"""

    def __init__(self, window: int):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.win: deque[float] = deque(maxlen=window)

    def push(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        self.win.append(x)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / max(1, self.n - 1)) if self.n > 1 else 1.0

    def z(self, x: float) -> float:
        return (x - self.mean) / max(self.std, 1e-9)

    def quantile(self, q: float) -> float:
        if not self.win:
            return float("inf")
        s = sorted(self.win)
        return s[min(len(s) - 1, int(q * len(s)))]

    def density(self, q: float) -> float:
        """滑动窗内超过 q 分位值的比例（惊讶密度）。"""
        if not self.win:
            return 0.0
        tau = self.quantile(q)
        return sum(1 for x in self.win if x >= tau) / len(self.win)


class Gates:
    """从特征算门控值。无状态部分查表，有状态部分（惊讶统计）自持。"""

    def __init__(self, params: GateParams):
        self.p = params
        self.stats = RunningStats(params.window)

    # ---- 写入门 ----
    def should_write(self, surprise: float) -> bool:
        if len(self.stats.win) < 8:      # 冷启动：统计未稳，先写
            return True
        return surprise >= self.stats.quantile(self.p.tau_quantile)

    # ---- 三门 ----
    def theta(self, surprise: float) -> float:
        z = self.stats.z(surprise)
        t = self.p.theta_base + (self.p.theta_max - self.p.theta_base) / (1 + math.exp(-z))
        return min(t, self.p.theta_clip)

    def eta(self) -> float:
        d = self.stats.density(self.p.tau_quantile)
        return self.p.eta_min + (self.p.eta_max - self.p.eta_min) * d

    def alpha(self, redundancy: float) -> float:
        return self.p.alpha_base + self.p.alpha_r * max(0.0, min(1.0, redundancy))

    def observe(self, surprise: float) -> None:
        self.stats.push(surprise)
