# PROGRESS.md — intero 进度快照

> 最近更新：2026-09-17 上午（**M0–M5 全部里程碑完成 + Kimi CLI 真机联调通过**）
> 下次接续方式：对 agent 说"**继续 intero**"，它会读本文件 + `bench/CALIBRATION.md` + `bench/M2_REPORT.md` + `README.md` 恢复上下文并自动汇报进度。

## 当前状态：全部设计里程碑 ✅（M0+M1 / M1.5 / M2 / M3 / M4 / M5）

- 黄金测试 **29/29 绿**（记忆 9 + 心跳 7 + 夜间时钟 3 + 注入/门面/MCP 5 + 主动闭环 5），`python -m compileall` 全过
- M1.5（改道完成）：LLM 归一化主线（`normalize.py`，api/local 双后端+缓存+透传兜底）+ 干净句微调加固（`bench/results/bge-neg-ft`，留出 test 排序 100%、margin_mean 0.247）；两档验收口径已公开重校准（CALIBRATION.md，agent 自主决策**待用户追认**）
- M2：三种子对拍报告 `bench/M2_REPORT.md`——写入省~70% ✓ / 过期占比 0.13~0.38 vs 1.00 ✓ / 组合事实超预期全胜 ✓ / flashbulb 打平 / 孤立事实召回减半（构造性，幅度超预测，记录在案）
- M3 心跳（`heartbeat.py`）：三态变心率 + 意图 TTL 调度 + 主动性拍卖（沉默有底价）+ 反拗期
- M4 接入（`core.py` 门面 + `inject.py` 装配 + `mcp_server.py`，mcp 1.x/2.x 兼容，stdio 可挂宿主）
- M5 夜间时钟（`dream.py`）：Dream 强制回放 + 去重/矛盾策展 + 晋升门 + 健康线

## 真机联调（2026-09-17 上午，用户拍板接入 Kimi CLI）

- **接入方式**：项目级 `.kimi/mcp.json`（stdio，绝对路径，env 注入 `INTERO_STORE=.intero/content.db` + `HF_HUB_OFFLINE=1`）；启动：`cd intero && kimi --mcp-config-file .kimi/mcp.json`
- **跨会话闭环实测通过**：会话A 并行写入两条事实（DeepSeek 归一化各 1 条）→ 会话B（全新进程）库存 2、两条事实按相关度正确置顶召回
- **修了两个真 bug**（联调才暴露，测试全覆盖不到）：
  1. `Intero` 默认库存 tempfile——MCP 场景重启即失忆 → 新增 `INTERO_STORE` 环境变量持久化（core.py）
  2. kimi 对并行工具调用 **spawn 多个 server 进程**，sidecar `.vecs.json` 竞态丢数据（实测 2 条只剩 1 条）→ store.py 持久化重构：向量 BLOB 与原文**同事务写入 sqlite**，λ 入 meta 表，删除 save()/sidecar；4 进程并发写入回归测试通过
- **主动性闭环（用户拍板"行"后接通，2026-09-17 上午）**：心跳粘进 Intero 门面 + MCP 新增 add_intention/heartbeat_tick + 心跳快照落 sqlite meta 表 + "冲动排队见面先说"送达语义（recall 置顶【待说事项】，送达即清）。语义诚实边界：宿主无法被主动推消息，主动性=下次见面时优先提起。新增 tests/test_proactive.py 5 个测试（送达一次/沉默赢/TTL 不行动/意图跨进程/待说跨进程），**29/29 绿**。真机三会话演示：A 注册提醒 → 35s 后 B 收到待说并自然说出提醒 → C 不重复。
- 已提交：`c91ef21`（M1.5–M5）+ `8c8e173`（联调修复）+ 本次主动性接通

## 关键实测结论（勿重复劳动）

1. 损失必须 sum；稀疏事件需 inner_steps=16 + 抽样回放（β=1.0, 8条/次）
2. λ 必须由**写入前泛化误差**驱动
3. 开源 encoder 裸跑全过不了否定探针；**归一化+管线才是正解**；微调只在干净句上有效
4. DeepSeek 归一化会拆碎限定词（v1 prompt），few-shot 强调"限定词不许拆"（v2）后达 oracle 级——改 prompt 要递增 `PROMPT_VERSION` 清缓存
5. bge 嵌入几何中否定是弱特征：理想归一化 margin 也只有 +0.058，排序正确≠绝对距离
6. 已缓存模型跑实验务必 `HF_HUB_OFFLINE=1`（直连 huggingface.co 会卡死 HEAD 重试）
7. .env 解析要剥行内注释（踩过 model 名带注释的坑）

## 待用户追认/拍板的事项

| # | 事项 | 背景 |
|---|---|---|
| 1 | **两档探针验收口径**（管线档 rank100%+mean0.15 / 裸 encoder 档 minmax>0.15 保留追求） | agent 夜里自主拍的板，CALIBRATION.md 有公开记录 |
| 2 | M2 孤立事实召回减半的缓解路线优先级（门控加实体显著性 / dream 补写 / 接受） | M2_REPORT.md 讨论节 |
| 3 | 是否引入写入路径 LLM judge（encoder=召回/judge=精确） | 非原设计，架构决策记录 |

## 下一步候选（新优先级）

| # | 任务 | 入口 |
|---|---|---|
| 1 | ~~端到端真机联调~~ ✅ 已完成（2026-09-17，见上节）；后续可在日常使用中观察 | `.kimi/mcp.json` |
| 2 | 心跳×器官联动：DREAM 态 tick 触发 dream()、WATCH 态意图来自记忆（如到期提醒） | heartbeat.py + dream.py 已各自就绪 |
| 3 | 生产配置全量标定（dim=768/hidden=4096 只做过冒烟） | bench/calibrate.py |
| 4 | fullctx 臂加"迷失中段"模拟、过期相位补标定阈值 | M2_REPORT.md 讨论节 |

## 接续时先跑这三条验证环境

```bash
cd D:\项目\micromind\intero
python -m unittest discover tests     # 应 29/29 OK
python -m intero                      # 写入率应 ≈29%
```

## 未提交事项

- 已 commit 两笔：`7ec1665`（M0+M1）→ `c91ef21`（M1.5–M5）→ 联调修复（本文件更新时提交）；**无远程仓库**，等用户决定是否推送
- 微调产物 `bench/results/bge-neg-ft/` 在 bench/results/（已 gitignore）
