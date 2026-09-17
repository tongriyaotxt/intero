# intero

**让 agent 拥有主动找你的能力。**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen)](tests/)
[![MCP](https://img.shields.io/badge/MCP-server-purple)](.kimi/mcp.json.example)

[English README](README.md)

今天的 agent 全都只会等你开口。**intero 给任何冻结 LLM 一条自主性循环**——
裁决"何时该说"的心跳、让它"想说"的内态驱动、以及让开口"有内容"的记忆。底座一个参数不动。

- 🧠 **记忆器官**——归一化原子事实、显著性门控写入、夜间 dream 策展、矛盾仲裁。重启不失忆；可导出成人可读 wiki 审计。
- 💓 **心跳器官**——自主性循环：意图在拍卖里与"沉默底价"竞价，内态驱动（孤独/好奇/记忆健康）没事也会起心动念，daemon 通过 Windows 气泡/语音主动找你。

```
17:47  （你不在）  🎈 弹窗："今晚还去攀岩不？"
17:49  （你打开 LLM CLI，全新会话）
       【它曾主动说】09-17 17:47 它说：今晚还去攀岩不？
       LLM 捡起线头，带着完整记忆上下文继续唠。
```

## "主动找你"到底是什么意思

不是定时任务，不是通知规则，是一套有刹车的判断系统：

- **与沉默竞价**——每条意图都要出价压过"沉默底价"才准开口，弱冲动自动输掉；
- **熟成曲线**——死线越近出价越高，过了半衰期衰减，到 TTL 作废；
- **内态驱动**——孤独（24h 没聊）、好奇（信息饥渴）、记忆健康（矛盾积压），零外部事件也会起心动念；
- **反拗期与作息底价**——说完话必须冷却；23 点到早 7 点底价抬到 0.9，深夜不叫醒；
- **理睬账本**——它记录你理没理，总被无视的话题自动闭嘴；
- **对话线头**——主动说过的话出现在你下次会话的上下文里（【它曾主动说】），直接接着唠。

背后支撑的是记忆器官：归一化原子事实、显著性门控写入、夜间 dream 策展、矛盾仲裁、人可读 wiki 导出。

## 为什么不用纯 RAG？

对拍基准，14 天模拟生活（630 条摄入），`bench/longitudinal.py` 可复跑：

| | hit@5 | 库存 | 备注 |
|---|---|---|---|
| **intero（light）** | **0.977** | 626 | 策展：553 晋升 / 17 去重 / 16 矛盾对标记 |
| 朴素 RAG（全写） | 0.977 | 630 | 无策展，矛盾悄悄堆积 |
| intero（Titans full） | 0.98 | 482 | 惊讶门控；λ 全程 0.0——见诚实标注 |

外加 RAG 结构性做不到的事：过期遗忘、矛盾浮现、**主动开口**。

## 快速开始（Windows）

```bat
git clone https://github.com/tongriyaotxt/intero.git
cd intero
scripts\setup.bat       :: venv + 依赖 + encoder 预下载 + 自检
scripts\chat.bat        :: 单开新窗口，Kimi CLI 挂上 intero MCP
```

Linux/macOS：`bash scripts/setup.sh`，然后起守护进程：

```bash
INTERO_STORE=.intero/content.db PYTHONPATH=. python -m intero.daemon --serve
```

## 架构

```
宿主 LLM（冻结）── MCP stdio ──► mcp_server（瘦客户端）
                                    │ localhost HTTP（~50ms）
                      ┌─────────────▼─────────────┐
                      │  daemon（常驻心脏起搏器）    │
                      │   Intero 门面              │
                      │   ├─ ContentStore（sqlite：原文+向量同事务）
                      │   ├─ 门控：显著 ∨ 新颖
                      │   ├─ Heartbeat：拍卖 / 反拗期 / 作息底价
                      │   ├─ 内态：孤独 · 好奇 · 记忆健康
                      │   └─ dream 夜间周期：策展→意图自生→导出 wiki
                      └──────┬───────────┬────────┘
                        气泡/语音        对话内线头
                        （你不在）       【它曾主动说】（你回来）
```

**MCP 工具**：`memory_write` · `memory_recall` · `memory_status` · `add_intention` · `heartbeat_tick` · `dream_now`

适配任意 MCP 宿主。Kimi CLI：`kimi --mcp-config-file .kimi/mcp.json`（参考 `.kimi/mcp.json.example`）。

## 可拓展性

每个器官都是插座——换掉任何一个都不影响其他：

| 插座 | 默认实现 | 可以换成 |
|---|---|---|
| 宿主 LLM | 任意 MCP 宿主（Kimi CLI 已验证） | Claude Code、Cursor、你自己的 agent 循环 |
| Encoder | bge-base-zh（SentenceTransformers） | 任意嵌入 API/模型（`Encoder` 协议） |
| 归一化/话术/意图自生 LLM | DeepSeek（OpenAI 兼容） | Kimi、本地 vLLM、离线=透传 |
| 通知通道 | Windows 气泡 / SAPI 语音 | Telegram/IM 机器人——实现一个函数即可 |
| 内态驱动 | 孤独 / 好奇 / 记忆健康 | 加一个驱动 = 一个字典项 + 一个指标函数 |
| 记忆核心 | light（显著∨新颖） | `INTERO_MODE=full` 启用 Titans 在线权重 |

daemon 只是 `Intero` 外面的一层循环；MCP server 只是 HTTP 上的一层薄壳。
所有状态住在一个 sqlite 文件里——可备份、可快照、可一键遗忘。

## 关键思想（全部实测，诚实汇报）

- **LLM 归一化 > 微调 encoder**：开源中文 encoder 全员败于整句否定（"我从来不喝咖啡"≈"我喝咖啡不加糖"，余弦 0.78）。写入前把原话改写成原子事实，陷阱整体消失，且人可审计。
- **写入率=召回天花板**：玩具规模测不出来；生活规模下纯惊讶门 hit@5 崩到 0.12。修复：显著∨新颖门 → 0.98。完整记录 [bench/LONGITUDINAL.md](bench/LONGITUDINAL.md)。
- **主动性必须有刹车**：沉默有底价、行动进反拗期、理睬账本（你理没理）会自动给总被无视的话题降温。
- **一切状态抗进程死亡**：MCP 宿主 spawn-per-call——记忆/心跳/意图/账本全部住在一个 sqlite 里。

## 诚实边界

- Titans 在线权重（`INTERO_MODE=full`）保留为实验层；在我们的条件下（冻结 encoder+稀疏事件+个人规模）参数化读出从未跑赢随机水平（λ≡0），**light 是默认模式**。这不是否定 Titans，是我们约束下的负面结果。
- 显著性 v1 是用户自指正则；真实混杂语料会漏（v2 方向：LLM 重要性打分）。
- 单用户；通知通道 Windows 优先（气泡/SAPI，其他系统欢迎 PR）。
- LLM 功能（归一化/意图自生/话术）需要 OpenAI 兼容 API key（`.env`，见 `.env.example`）；离线自动降级为原文透传。
- Windows 气泡在专注助手开启时可能被静默——见 notification center 或检查系统通知设置。

## 路线图

- [ ] 对话消化 → 更准的追问/关心类意图
- [ ] wiki 升级为主存储（当前只读导出）——真正的可删除权
- [ ] 送达通道：Telegram/IM 机器人
- [ ] 多用户隔离

## 许可证

Apache-2.0。接入的模型权重（如 bge）自带各自许可证。
