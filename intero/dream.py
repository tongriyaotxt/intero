"""dream.py — M5 夜间时钟：Dream 回放 + 策展 → 巩固 → 晋升门

夜间时钟是"睡眠巩固"的 sidecar 版（tech-base/23 的快权重→慢权重蒸馏，
这里巩固的受体是 Titans 快权重本身 + 内容库的整理）：

  Dream 回放：睡眠无预算压力——对存活条目强制写入（绕过门控、加深内循环），
             把白天门控漏掉的埋点补进参数化记忆；
  策展：     近重复条目去重（旧的归档为 dedup 不再参与检索），
             疑似矛盾对标记为 conflict（不删——矛盾事实都可能是真的，读给 LLM 裁决）；
  晋升门：   记忆表示良好的事实（自重构误差低于中位数）晋升 kind="consolidated"，
             噪声永不晋升——"记得牢的"和"只是见过的"从此分层；
  健康线：   回放后自重构误差不得恶化（恶化说明回放配置有毒，报告标红）。

红线：dream 只动 sidecar 内部状态；晋升/归档都保留原文，操作可审计可回退。
"""

from __future__ import annotations

import numpy as np

from .memory import TitansMemory
from .store import ContentStore

DEDUP_COS = 0.97        # 近重复判重阈值
CONFLICT_LOW, CONFLICT_HIGH = 0.75, 0.97   # 疑似矛盾带（像又没那么像）


def dream(
    mem: TitansMemory,
    store: ContentStore,
    replay_inner_steps: int = 24,
    verbose: bool = False,
) -> dict:
    """跑一个夜间周期，返回报告。"""
    report: dict = {"回放": 0, "去重": 0, "矛盾对": 0, "晋升": 0}
    items = store.items()
    if not items:
        return report

    recon_before = mem.recon_error()

    # ---- Dream 回放：强制写入，加深内循环 ----
    saved_steps, saved_early = mem.inner_steps, mem.early_stop_mse
    mem.inner_steps = replay_inner_steps
    for it in items:
        if it["kind"] == "noise":
            continue
        mem.write(it["vec"], it["vec"], redundancy=store.redundancy(it["vec"]), force=True)
        report["回放"] += 1
    mem.inner_steps, mem.early_stop_mse = saved_steps, saved_early

    # ---- 策展：去重 + 矛盾标记 ----
    V = np.stack([it["vec"] for it in items])
    sims = V @ V.T
    archived: set[int] = set()
    for i in range(len(items)):
        if items[i]["id"] in archived:
            continue
        for j in range(i + 1, len(items)):
            if items[j]["id"] in archived:
                continue
            s = sims[i, j]
            if s >= DEDUP_COS:
                store.update_kind(items[j]["id"], "dedup")   # 后来者归档
                archived.add(items[j]["id"])
                report["去重"] += 1
            elif CONFLICT_LOW <= s < CONFLICT_HIGH and items[i]["kind"] == items[j]["kind"] == "fact":
                store.update_kind(items[i]["id"], "conflict")
                store.update_kind(items[j]["id"], "conflict")
                report["矛盾对"] += 1

    # ---- 晋升门：记忆表示良好（重构误差低于中位数）的事实晋升 ----
    survivors = [it for it in store.items() if it["kind"] in ("fact", "composite")]
    if survivors:
        errs = []
        for it in survivors:
            m = mem.read(it["vec"])
            errs.append(float(((m - it["vec"]) ** 2).mean()))
        median = float(np.median(errs))
        for it, e in zip(survivors, errs):
            if e <= median:
                store.update_kind(it["id"], "consolidated")
                report["晋升"] += 1

    report["自重构误差"] = {"前": recon_before, "后": mem.recon_error()}
    if recon_before is not None and mem.recon_error() is not None:
        report["健康"] = mem.recon_error() <= recon_before * 1.5
    if verbose:
        print("[dream]", report)
    return report
