# intero

> 给任意冻结大模型接上两个体外器官：**会学习的记忆**（Titans sidecar）和**会自发的心跳**（autonomy loop）。
> 论文依据：[arXiv:2501.00663 Titans](https://arxiv.org/abs/2501.00663)；设计讨论：工作区 tech-base/21、22。
> 当前状态：**竖片（M0+M1）完成**——记忆核心 + 内容库 + 标定 + 冒烟 demo。

## 它是什么

一个即插即用的神经长期记忆：底座 LLM 一个参数不动，sidecar 在测试时做**真实的梯度更新**：

```
ℓ = ||M(k)−v||²            联想记忆损失   (论文 Eq.12)
S_t = η·S_{t−1} − θ·∇ℓ     过去惊讶动量   (论文 Eq.14) → flashbulb 记忆
M_t = (1−α)·M_{t−1} + S_t  遗忘门         (论文 Eq.13)
y = M(q)                   读出，前向不更新 (论文 Eq.15)
```

- **惊讶门控写入**：瞬时惊讶 = ‖∇ℓ‖，running 分位数门控 → 省 ~70% 写入（构造保证）
- **手工门控**（θ/η/α 各吃一个语义特征）：对论文学习门的工程替代
- **在线回放**：新记忆不盖旧记忆（防顺序写入互相覆盖）
- **快照回滚**：自重构误差恶化 → 自动回滚，"部署即学习"不变成"部署即腐烂"
- **双路读出**：score = λ·cos(M(q),v) + (1−λ)·cos(q,v)；λ 由**写入前泛化误差**驱动（不是训练集重构误差——那会把 λ 骗到上限），学不到结构时 λ→0 自动退化为纯 RAG，永不帮倒忙

## 快速开始

```bash
pip install -e .            # 依赖仅 torch + numpy
python -m unittest discover tests   # 黄金测试（24 个：记忆/门控/心跳/夜间时钟/注入）
python -m intero                  # 冒烟 demo：100事实+300噪声→10问
PYTHONPATH=. python -m intero.mcp_server   # MCP server（stdio，宿主可挂）
```

## 代码地图

| 文件 | 职责 |
|---|---|
| `intero/memory.py` | Titans 记忆核心（Eq.12–14 手写更新，不用 torch.optim） |
| `intero/gates.py` | 手工门控 θ/η/α + 写入门（冠军超参已冻结，见 bench/CALIBRATION.md） |
| `intero/store.py` | 内容库（**永远存原文**——换 encoder 可重建）+ 双路读出 λ 混合 |
| `intero/encoder.py` | 可插拔 encoder：bge(ST) → API → TF-IDF 兜底；中文探针 margin>0.15 准入 |
| `intero/normalize.py` | 写入前 LLM 归一化（api/local 双后端 + 磁盘缓存 + 透传兜底） |
| `intero/core.py` | Intero 门面：记忆器官整机（M4 接入载体） |
| `intero/inject.py` | prompt 注入装配（consolidated 优先 / conflict 标注 / 预算截断） |
| `intero/heartbeat.py` | M3 心跳：三态变心率 + 意图调度 + 主动性拍卖 + 反拗期 |
| `intero/dream.py` | M5 夜间时钟：Dream 回放 + 策展 + 晋升门 |
| `intero/mcp_server.py` | MCP server（memory_write/recall/status 三工具） |
| `bench/calibrate.py` | 三相位标定台（背记/抗噪/过期），冠军配置来源 |
| `bench/CALIBRATION.md` | 标定实录（失败诊断 + 冻结值 + 探针实录 + M1.5 改道） |
| `bench/scenarios.py` / `m2_benchmark.py` | M2 剧本生成器 + 三方对拍台 |
| `bench/M2_REPORT.md` | M2 三种子对拍报告（对照预先登记表） |
| `bench/probe_step0.py` / `probe_pipeline.py` / `probe_oracle.py` | 探针实验组（零训练候选 / 管线验收 / 理想归一化隔离） |
| `bench/finetune_encoder.py` | 干净句对比微调（扩否定 margin） |

## 对论文的偏离（诚实标注）

1. **门控 α/η/θ 是手工函数不是学出来的**——底座冻结，没有 outer loop；用标定台一次性搜索+冻结补偿；
2. **嵌入不回传梯度**——k/v 来自冻结 encoder（encoder 质量是系统天花板，探针准入制）；
3. **读出走 ANN 回原文**——参数化泛化体现在路由，λ 爬升幅度即其量化；
4. **稀疏事件迭代**——论文逐 token 单步更新；我们每个事件迭代 `inner_steps` 步 + 在线回放，补偿稀疏性；
5. **回放缓冲**——论文无此组件（它有动量），我们的稀疏场景需要显式防遗忘。

## 路线图

- [x] M0+M1 竖片：记忆核心 + 内容库 + 标定 + 冒烟 demo（demo 写入率 29%，TF-IDF 下 λ→0 优雅退化）
- [x] **M1.5 encoder 攻坚（改道完成）**：零训练手段全灭（Step 0 实录）→ **LLM 归一化**主线上位（`intero/normalize.py`）+ 干净句轻量微调加固（留出 test 排序 100%、margin_mean 0.135→0.247）；两档验收口径公开重校准（CALIBRATION.md）
- [x] M2 基准：剧本生成器三类埋点+噪声+过期事实，三种子三方对拍 → **报告 [bench/M2_REPORT.md](bench/M2_REPORT.md)**（写入省~70% ✓、过期占比 3~8 倍优势 ✓、组合事实超预期全胜 ✓、孤立事实召回减半⚠️ 记录在案）
- [x] M3 心跳：三态变心率 + 意图调度器（TTL）+ 主动性拍卖（沉默是竞拍者）+ 反拗期（`intero/heartbeat.py`，测试 7/7）
- [x] M4 接入：Intero 门面（`core.py`）+ prompt 注入装配（`inject.py`）+ MCP server（`mcp_server.py`，mcp 1.x/2.x 兼容）
- [x] M5 夜间时钟：Dream 回放 + 策展去重/矛盾标记 → 晋升门（`dream.py`，测试 3/3）

## 预先登记的胜负预测（M2 不许事后改口）

| 指标 | 预测 |
|---|---|
| 写入量 vs RAG 全存 | 🟢 赢 ~70%（构造保证） |
| 过期内容占比 | 🟢 赢（RAG 无遗忘机制） |
| flashbulb 集群细节 | 🟡 小赢到打平 |
| 组合事实召回 | 🟡 打平（λ 爬升幅度即测量值） |
| 孤立事实召回 | 🔴 可能小输（门控漏埋点的代价） |
