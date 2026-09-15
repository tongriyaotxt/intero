"""memory.py — Titans 神经记忆：测试时梯度更新（Eq.12–14 忠实实现）。

论文对照（arXiv:2501.00663 §3.1）：
  ℓ(M; x_t) = ||M(k_t) − v_t||²            (Eq.12  联想记忆损失)
  S_t = η_t·S_{t−1} − θ_t·∇ℓ(M_{t−1}; x_t) (Eq.14  过去惊讶动量)
  M_t = (1−α_t)·M_{t−1} + S_t              (Eq.13  遗忘门)
  读出 y = M(q)，前向不更新                 (Eq.15)

保险（论文靠 outer loop 学到的稳定性，我们手工门控没有，必须工程补齐）：
  1. grad norm clip + θ 上限 clamp；
  2. 每 snapshot_every 次写入存快照，rolling 自重构回测恶化即回滚——
     "部署即学习"不变成"部署即腐烂"的关键。
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .gates import Gates, GateParams


class MemoryMLP(nn.Module):
    """深记忆（论文 §5.5 消融：深 > 线性）。SiLU 激活，末层小初始化。"""

    def __init__(self, dim: int, hidden: int, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TitansMemory:
    def __init__(
        self,
        dim: int = 768,
        hidden: int = 4096,
        depth: int = 3,
        gate_params: GateParams | None = None,
        snapshot_every: int = 16,
        probe_window: int = 32,
        rollback_factor: float = 3.0,
        inner_steps: int = 16,
        replay_size: int = 128,
        replay_beta: float = 1.0,
        replay_sample: int = 8,
        theta_decay: float = 1.0,
        early_stop_mse: float | None = 1e-3,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.model = MemoryMLP(dim, hidden, depth)
        self.params = list(self.model.parameters())
        # S：与参数同构的动量缓冲（Eq.14 的 S_t）
        self.S = [torch.zeros_like(p) for p in self.params]
        self.gates = Gates(gate_params or GateParams())
        self.snapshot_every = snapshot_every
        self.rollback_factor = rollback_factor
        self.writes = 0
        self.rollbacks = 0
        self.inner_steps = inner_steps  # 稀疏事件写入的迭代次数（偏离#4，见 README）
        self.replay_beta = replay_beta  # 在线回放混合比（防顺序写入互相覆盖）
        self.replay_sample = replay_sample  # 每次写入抽样的回放条数（全量回放太贵）
        self.theta_decay = theta_decay    # 内循环步长衰减（抑制动量过冲）
        self.early_stop_mse = early_stop_mse  # 每维均方误差低于此值提前停（省算力防过冲）
        self._replay: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=replay_size)
        self._rng = np.random.default_rng(seed)
        self._snapshots: deque[dict] = deque(maxlen=2)
        self._probe_buf: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=probe_window)
        self._probe_errs: deque[float] = deque(maxlen=probe_window * 4)
        self.last_surprise: float = 0.0
        self.last_grad_norm: float = 0.0
        self.last_prewrite_mse: float | None = None

    # ---- 内部 ----

    def _loss(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # sum（=mean×dim）：v 单位归一化时 mean 损失 ~1/dim，梯度消失；sum 保持有效步长
        return ((self.model(k) - v) ** 2).sum()

    def _grads(self, loss: torch.Tensor) -> tuple[list[torch.Tensor], float]:
        grads = torch.autograd.grad(loss, self.params)
        total = torch.sqrt(sum((g ** 2).sum() for g in grads)).item()
        clip = self.gates.p.grad_clip
        if total > clip > 0:
            scale = clip / (total + 1e-12)
            grads = [g * scale for g in grads]
        return list(grads), total

    @staticmethod
    def _t(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    # ---- Eq.8：瞬时惊讶 = ||∇ℓ||（不更新） ----

    def surprise(self, k: np.ndarray, v: np.ndarray) -> float:
        loss = self._loss(self._t(k), self._t(v))
        _, gn = self._grads(loss)
        self.last_surprise = gn
        return gn

    # ---- Eq.12–14：写入 ----

    def write(self, k: np.ndarray, v: np.ndarray, redundancy: float = 0.0, force: bool = False) -> bool:
        """返回是否实际执行了更新（写入门控可能拒绝）。"""
        kt, vt = self._t(k), self._t(v)
        loss = self._loss(kt, vt)
        grads, gn = self._grads(loss)
        self.last_grad_norm = gn
        self.last_prewrite_mse = loss.item() / kt.numel()  # 泛化探针：写入前损失≈对新样本的预测质量

        g = self.gates
        if not force and not g.should_write(gn):
            g.observe(gn)
            return False

        theta = min(g.theta(gn), g.p.theta_clip)
        eta = g.eta()
        alpha = g.alpha(redundancy)

        reps: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self._replay and self.replay_beta > 0:  # 写入前抽样一次，内循环复用
            buf = list(self._replay)
            idx = self._rng.choice(len(buf), size=min(self.replay_sample, len(buf)), replace=False)
            reps = [buf[i] for i in idx]
        dim = kt.numel()
        for i in range(self.inner_steps):  # 每个事件迭代多步（稀疏写入补偿）
            loss_i = self._loss(kt, vt)          # 要在 grad 模式下算图
            if self.early_stop_mse is not None and loss_i.item() / dim < self.early_stop_mse:
                break  # 已背下，提前停
            if reps:                             # 在线回放：新记忆不盖旧记忆
                loss_i = loss_i + self.replay_beta * sum(
                    self._loss(kr, vr) for kr, vr in reps
                ) / len(reps)
            grads_i, _ = self._grads(loss_i)
            theta_i = theta * (self.theta_decay ** i)  # 步长衰减
            with torch.no_grad():                # 只有参数更新禁梯度
                for p, s, gr in zip(self.params, self.S, grads_i):
                    s.mul_(eta).add_(gr, alpha=-theta_i)  # Eq.14: S ← η·S − θ·∇ℓ
                    p.mul_(1 - alpha).add_(s)             # Eq.13: M ← (1−α)·M + S
        self._replay.append((kt.detach().clone(), vt.detach().clone()))
        g.observe(gn)
        self.writes += 1

        self._probe_buf.append((kt.detach().clone(), vt.detach().clone()))
        if self.writes % self.snapshot_every == 0:
            self._snapshot()
            self._health_check()
        return True

    # ---- Eq.15：读出（前向不更新） ----

    def read(self, q: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model(self._t(q)).numpy()

    # ---- 保险：快照 + 自重构回测 + 回滚 ----

    def _snapshot(self) -> None:
        self._snapshots.append({
            "model": copy.deepcopy(self.model.state_dict()),
            "S": [s.clone() for s in self.S],
            "writes": self.writes,
        })

    def recon_error(self) -> float | None:
        """近期写入样本的自重构误差（健康探针）。"""
        if len(self._probe_buf) < 4:
            return None
        with torch.no_grad():
            errs = [((self.model(k) - v) ** 2).mean().item() for k, v in self._probe_buf]
        return float(np.mean(errs))

    def _health_check(self) -> None:
        e = self.recon_error()
        if e is None:
            return
        self._probe_errs.append(e)
        if len(self._probe_errs) < 3 or not self._snapshots:
            return
        baseline = float(np.median(list(self._probe_errs)[:-1]))
        if e > max(baseline * self.rollback_factor, baseline + 0.1):
            snap = self._snapshots[0]
            self.model.load_state_dict(snap["model"])
            for s, s_snap in zip(self.S, snap["S"]):
                s.copy_(s_snap)
            self.rollbacks += 1
            self._probe_errs.clear()

    # ---- 持久化（L3 连续性的载体） ----

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "S": self.S, "writes": self.writes}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, weights_only=False)
        self.model.load_state_dict(ck["model"])
        for s, s_ck in zip(self.S, ck["S"]):
            s.copy_(s_ck)
        self.writes = ck["writes"]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.params)

    @property
    def past_surprise_norm(self) -> float:
        """‖S‖：过去惊讶强度，flashbulb 判定的直接读数。"""
        with torch.no_grad():
            return torch.sqrt(sum((s ** 2).sum() for s in self.S)).item()
