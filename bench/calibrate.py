"""bench/calibrate.py — 超参标定台：把"调参"变成一次性搜索 + 冻结。

三相位协议（每个候选配置跑一遍）：
  相位1 背记：100 个随机 k-v 对强制写入 → 召回率 = mean cos(M(k_i), v_i)
  相位2 抗噪：写入门控开启灌 300 条噪声 → 旧记忆保持率 + 实际写入量
  相位3 过期遗忘：20 个键写入新值（"搬家了"）→ 新值召回高、旧值召回低

复合分 = 召回 + 保持 + 新值召回 − |旧值召回 − 0.3|（旧值应显著衰退）
冠军配置 → tests/test_memory.py 的黄金测试锁死，之后谁改谁举证。
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from intero.gates import GateParams
from intero.memory import TitansMemory

RNG = np.random.default_rng(42)


def rand_unit(n: int, dim: int) -> np.ndarray:
    x = RNG.normal(size=(n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run_phases(gp: GateParams, dim: int, hidden: int, depth: int, inner_steps: int, seed: int = 0) -> dict:
    mem = TitansMemory(dim=dim, hidden=hidden, depth=depth, gate_params=gp, inner_steps=inner_steps, seed=seed)

    K1, V1 = rand_unit(100, dim), rand_unit(100, dim)
    for k, v in zip(K1, V1):
        mem.write(k, v, force=True)
    recall = float(np.mean([mem.read(k) @ v for k, v in zip(K1, V1)]))

    K2, V2 = rand_unit(300, dim), rand_unit(300, dim)
    for k, v in zip(K2, V2):
        mem.write(k, v)
    retention = float(np.mean([mem.read(k) @ v for k, v in zip(K1, V1)]))
    noise_write_ratio = mem.writes / (100 + 300)  # 含相位1的 100 次强制写入

    # 相位3：前 20 个键"过期"（赋新值），后 80 个保持活跃
    V1_new = rand_unit(20, dim)
    for k, v in zip(K1[:20], V1_new):
        mem.write(k, v, force=True)
    for k, v in zip(K1[20:], V1[20:]):  # 活跃键再强化一次
        mem.write(k, v, force=True)
    stale = float(np.mean([mem.read(k) @ v for k, v in zip(K1[:20], V1[:20])]))
    fresh = float(np.mean([mem.read(k) @ v for k, v in zip(K1[:20], V1_new)]))

    score = recall + retention + fresh - abs(stale - 0.3)
    return {
        "recall": recall, "retention": retention, "stale": stale, "fresh": fresh,
        "noise_write_ratio": noise_write_ratio, "rollbacks": mem.rollbacks, "score": score,
    }


def main(n_trials: int = 24, dim: int = 128, hidden: int = 1024, depth: int = 3):
    rng = np.random.default_rng(7)
    results = []
    t0 = time.time()
    for i in range(n_trials):
        gp = GateParams(
            theta_base=float(10 ** rng.uniform(-2.5, -1.0)),
            theta_max=float(10 ** rng.uniform(-1.0, -0.3)),
            eta_min=float(rng.uniform(0.3, 0.7)),
            eta_max=float(rng.uniform(0.85, 0.98)),
            alpha_base=float(10 ** rng.uniform(-3.5, -2.0)),
            alpha_r=float(rng.uniform(0.005, 0.05)),
            theta_clip=0.35,
        )
        inner_steps = int(rng.choice([8, 16, 32]))
        r = run_phases(gp, dim, hidden, depth, inner_steps)
        r["params"] = gp.__dict__ | {"inner_steps": inner_steps}
        results.append(r)
        print(f"[{i+1}/{n_trials}] score={r['score']:.3f} recall={r['recall']:.3f} "
              f"ret={r['retention']:.3f} stale={r['stale']:.3f} fresh={r['fresh']:.3f} "
              f"wr={r['noise_write_ratio']:.2f} rb={r['rollbacks']}")
    results.sort(key=lambda r: -r["score"])
    best = results[0]
    print(f"\n=== 冠军（{time.time()-t0:.0f}s, {n_trials} trials, dim={dim} hidden={hidden}）===")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    with open("bench/calibration_result.json", "w") as f:
        json.dump({"champion": best, "top5": results[:5]}, f, indent=2, ensure_ascii=False)
    print("已写 bench/calibration_result.json")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    main(n_trials=n)
