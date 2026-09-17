@echo off
rem setup.bat — intero 一键安装（Windows）
setlocal
cd /d "%~dp0\.."

echo [1/4] 创建虚拟环境并安装依赖...
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -e .[st,mcp]

echo [2/4] 配置 .env...
if not exist .env (
    copy .env.example .env
    echo   已创建 .env —— 请编辑填入你的 LLM API key（不填也能跑，自动降级）
) else (
    echo   .env 已存在，跳过
)

echo [3/4] 预下载 encoder（bge-base-zh，~200MB，仅首次）...
set HF_ENDPOINT=https://hf-mirror.com
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-zh-v1.5'); print('encoder ready')"

echo [4/4] 跑测试自检...
python -m pytest tests -q

echo.
echo === 完成 ===
echo   启动守护进程（主动搭话+常驻服务）: scripts\start_daemon.bat
echo   单开窗口和它聊天:                  scripts\chat.bat
echo   冒烟 demo:                         python -m intero
endlocal
