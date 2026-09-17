# intero

> 给任意冻结大模型接上两个体外器官：**记忆**（内容库 + 门控 + 夜间策展）和**会自发的心跳**（autonomy loop）。
> 当前状态：**全部里程碑完成 + 服务化 + 主动性闭环 + 纵向模拟验证**（详见 PROGRESS.md）。

> **架构决策（2026-09-17，用户拍板）**： Titans 在线权重降级为 `full` 实验层，默认 **light 模式**——
> 依据：纵向模拟 λ 全程 0.0（参数化读出在生活规模未兑现，bench/LONGITUDINAL.md）。
> light 门控 = 显著（关乎用户）∨ 新颖（冗余度<阈值）；召回行为与 full 相同（λ 本来为 0）。
> 切换：`INTERO_MODE=full` 或构造时传 `memory=`。完整实证脉络见 tech-base/24。

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

### 主动搭话守护进程（daemon）

MCP 是被动应答（宿主调用才活）；`daemon` 是常驻起搏器——自己跳心跳，拍卖赢出的冲动**主动找你**，
并内嵌常驻 HTTP 服务（`--serve`，只绑 127.0.0.1）让 MCP 瘦客户端毫秒级调用：

```bash
cd intero
INTERO_STORE=.intero/content.db HF_HUB_OFFLINE=1 PYTHONPATH=. \
    .venv/Scripts/python.exe -m intero.daemon --serve      # 气泡通知；加 --voice 语音播报
# 或一键：scripts\start_daemon.bat（开机自启：手动运行 scripts\install_autostart.bat）
# 日常唠嗑入口：scripts\chat.bat——单开新终端窗口起 kimi（独立上下文），
#               daemon 没在跑会先自动拉起
```

服务化实测：recall 47~140ms vs spawn 冷启动 ~15s（>100×）；MCP 在服务不在时回退本地实例。

送达路由：你在聊（ENGAGED，60 秒内有交互）→ 冲动留给对话内 recall 送达，不打扰；
你离开（WATCH/DREAM）→ Windows 气泡（+可选 SAPI 语音）主动搭话。
daemon 每拍前先 reload sqlite 快照（吸收 MCP 侧新意图），跳完即存——进程间只通过库通信。
已实测（2026-09-17）：注册意图后 daemon 在 60 秒静默期结束的第一拍主动弹窗，送达即清不重复。
进入 DREAM（休眠）的第一拍自动跑夜间周期：dream 回放/策展/晋升 + **意图自生**（LLM 通读库存
事实，按 remind/followup/care/association 四类萌发意图，可审计可去重）。也可 MCP 调 `dream_now` 手动触发。
**内态驱动**（drives.py）：没事也会想起你——social（24h 没聊满分）/curiosity（摄入饥渴）/
memory_health（矛盾积压）三维内态超阈即起心动念，冷却 4h，冲动仍走正常拍卖（tech-base/23 落地）。
**作息底价**：23-7 点沉默底价抬到 0.9，深夜不叫醒。
**话术人格化**：弹窗前过一遍 LLM 润色（离线原文透传），润色成品回写记录。
**对话线头**：主动说过的话存 recent_said（48h 内），下次对话 recall 顶部【它曾主动说】——能接着唠。
**理睬反馈闭环**：主动送达记账，10 分钟内用户响应记为理睬；同类意图满 3 次送达后，
紧迫度乘子 = 0.5 + 理睬率——总被无视的话题自动闭嘴。
**实体显著性门控**：关于用户的事实绕过惊讶门直接写（纵向模拟实测：纯惊讶门控在生活规模下
hit@5 仅 0.12，写入率=召回天花板，见 bench/LONGITUDINAL.md）。

`.kimi/mcp.json` 已配好（stdio，绝对路径，记忆持久化到 `.intero/content.db`）：

```bash
cd intero
kimi --mcp-config-file .kimi/mcp.json   # 交互模式；工具：memory_write / memory_recall / memory_status / add_intention / heartbeat_tick
```

已实测跨会话闭环（2026-09-17）：会话A 并行写入两条事实 → 会话B（新进程）正确召回。
已实测主动性闭环（2026-09-17）：会话A 注册提醒意图 → 35 秒后会话B 的 recall 顶部收到
【待说事项】→ 会话C 已清除不重复。工具全表：memory_write / memory_recall / memory_status /
add_intention / heartbeat_tick。
**注意**：kimi 对并行工具调用会 spawn 多个 server 进程，存储层已改为 sqlite 单文件
（向量作 BLOB 与原文同事务写入）以抗竞态——不要用 sidecar 文件存向量。
心跳快照（意图/待说/反拗）存在同一 sqlite 的 meta 表，跨进程不丢。

**主动性语义（请求驱动宿主下的诚实形态）**：冲动排队，见面先说。每次工具调用 =
interact() + tick()（懒惰心跳，TTL/状态迁移走墙钟）；拍卖赢出的意图进 pending，
下次 recall 置顶送达并清除。宿主无法被主动推消息——这是当前架构的边界，不是 bug。

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
| `intero/heartbeat.py` | M3 心跳：三态变心率 + 意图调度 + 主动性拍卖 + 反拗期 + 待说队列 + 快照持久化 |
| `intero/dream.py` | M5 夜间时钟：Dream 回放 + 策展 + 晋升门 |
| `intero/mcp_server.py` | MCP server（memory_write/recall/status + add_intention/heartbeat_tick 五工具） |
| `intero/daemon.py` | 主动搭话守护进程：常驻起搏 + 气泡/语音通知 + DREAM 态夜间周期 |
| `intero/intents.py` | 意图自生：LLM 从库存事实抽取到期事项（api/null 双后端，失败安全空） |
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
