#!/usr/bin/env bash
# setup.sh — intero 一键安装（Linux/macOS）
set -e
cd "$(dirname "$0")/.."

echo "[1/4] venv + 依赖..."
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[st,mcp]"

echo "[2/4] .env..."
[ -f .env ] || { cp .env.example .env; echo "  已创建 .env（不填 API key 也能跑，自动降级）"; }

echo "[3/4] 预下载 encoder（仅首次）..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-zh-v1.5'); print('encoder ready')"

echo "[4/4] 测试自检..."
python -m pytest tests -q

cat <<'EOF'

=== 完成 ===
  守护进程:  INTERO_STORE=.intero/content.db PYTHONPATH=. python -m intero.daemon --serve
  MCP 宿主:  用 .kimi/mcp.json 接入 Kimi CLI，或把 intero.mcp_server 挂进任意 MCP 宿主
  冒烟 demo: python -m intero
EOF
