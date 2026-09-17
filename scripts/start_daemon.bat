@echo off
rem start_daemon.bat — 手动启动 intero 守护进程（常驻服务 + 主动搭话）
rem 用法：双击或命令行运行；停止 = 关窗口或 Ctrl+C
cd /d "%~dp0\.."
set INTERO_STORE=%CD%\.intero\content.db
set HF_HUB_OFFLINE=1
set PYTHONPATH=%CD%
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m intero.daemon --serve
