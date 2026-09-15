# PROGRESS.md — intero 进度快照

> 最近更新：2026-09-15（竖片完成）
> 下次接续方式：对 agent 说"**继续 intero**"，它会读本文件 + `bench/CALIBRATION.md` + `README.md` 恢复上下文并自动汇报进度。

## 当前状态：M0+M1 竖片 ✅ 完成

- 记忆核心（Eq.12–14 + 在线回放 + 快照回滚）、手工门控（冠军超参已冻结）、内容库（λ 双路读出）、可插拔 encoder + 探针
- 黄金测试 **9/9 绿**；冒烟 demo：写入率 29%（门控 ✓）、λ→0 优雅退化（TF-IDF 下）
- 合成标定：背记路由 top3=0.90，抗噪后 0.63
- 依赖：系统 Python 3.13 有 torch 2.13+cpu；`.venv/`（--system-site-packages）装有 sentence-transformers 6.0.1；bge-base-zh-v1.5 已缓存（HF_ENDPOINT=https://hf-mirror.com）

## 关键实测结论（勿重复劳动）

1. 损失必须 sum（mean 会梯度消失）；稀疏事件需 inner_steps=16 + 抽样回放（β=1.0, 8条/次）
2. λ 必须由**写入前泛化误差**驱动（训练集重构误差会把它骗到上限）
3. **bge 过不了否定探针**（margin −0.037，败于整句否定"我从来不喝咖啡只喝茶"0.778）；TF-IDF 与混合编码同样不过 → 激活 encoder 阶梯第 2 顺位

## 下一步（按优先级）

| # | 任务 | 入口 |
|---|---|---|
| 1 | **M1.5 encoder 微调**：用对比对（改写=正例，否定/翻转=负例）微调 bge-base-zh → 过探针（margin>0.15）→ 开源候选 | 训练对可由基准剧本生成器量产；sentence-transformers `InputExample` + `MultipleNegativesRankingLoss` 起步 |
| 2 | M2 基准：`bench/scenarios.py` 剧本生成器（孤立/组合/集群三类埋点 + 语义近邻噪声 + 过期事实）→ 三方对拍（sidecar vs RAG vs 全上下文），按 README 预先登记表验收 | 参考 `intero/__main__.py` 的 gen_corpus |
| 3 | M3 心跳：三态变心率 + 意图调度器 + 主动性拍卖（沉默是竞拍者）+ 反拗期 | 设计见对话记录（tech-base/21、22 风格） |
| 4 | M4 接入：prompt 注入装配 + MCP server | — |

## 接续时先跑这三条验证环境

```bash
cd D:\项目\micromind\intero
python -m unittest discover tests     # 应 9/9 OK
python -m intero                      # 写入率应 ≈29%
```

## 未提交事项

- 本项目未 git init（独立目录，待用户决定是否建仓库）
- 工作区根 `main_video分析报告.md`、tech-base/21、22、OpenHuman/docs/ADR-001 均已落盘但未 commit（两个子仓库各自独立）
