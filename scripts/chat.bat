@echo off
rem chat.bat — 单开一个新终端窗口，在 intero 目录起 kimi（带记忆+主动性 MCP）
rem 日常唠嗑入口：双击即可。与任何已有终端/会话完全隔离（独立上下文）。
rem 先确保 daemon 在跑（没跑则后台拉起一个隐藏窗口），再开聊天窗口。
cd /d "%~dp0\.."
set INTERO_STORE=%CD%\.intero\content.db
set HF_HUB_OFFLINE=1
set PYTHONPATH=%CD%
set PYTHONIOENCODING=utf-8

rem 若 7377 服务不在则后台拉起 daemon（--serve 常驻）
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:7377/health' -TimeoutSec 1) | Out-Null } catch { Start-Process -WindowStyle Hidden -FilePath '.venv\Scripts\python.exe' -ArgumentList '-m','intero.daemon','--serve' }"

rem 单开新窗口起 kimi（新会话、独立上下文）
start "intero chat" cmd /k "cd /d %CD% && set PYTHONIOENCODING=utf-8 && kimi --mcp-config-file .kimi/mcp.json"
