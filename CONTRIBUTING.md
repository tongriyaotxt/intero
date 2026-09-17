# 贡献指南

欢迎 PR。几条项目铁律（违反会被要求修改）：

1. **诚实标注**：失败、负面结果、无法复现的点必须写进文档，不许藏。
2. **测试**：`python -m pytest tests -q` 必须全绿；新功能带黄金测试（行为级，不测实现细节）。
3. **编码器无关**：不要在核心路径写死任何 encoder/LLM 供应商；一切外部依赖走可插拔接口 + 离线兜底。
4. **状态纪律**：所有状态进 sqlite（单文件），禁止 sidecar 文件（spawn-per-call 宿主下会竞态）。
5. **中文优先**：文档、注释、提交信息用简体中文（README 中英双语）。

## 开发

```bash
bash scripts/setup.sh      # 或 Windows: scripts\setup.bat
python -m pytest tests -q
python -m intero           # 冒烟 demo
```

## 基准

改动门控/检索逻辑前请先跑 `bench/longitudinal.py --run`（light 模式 ~3 分钟）确认无回归，
并把数字写进 PR 描述。
