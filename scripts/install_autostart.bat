@echo off
rem install_autostart.bat — 注册开机自启（登录时启动 intero daemon）
rem ⚠️ 这会修改你的系统（计划任务）。不需要就用 uninstall_autostart.bat 卸载。
rem 本脚本只在你手动双击/运行时生效，agent 不会替你执行。
schtasks /create /tn "intero-daemon" /tr "\"%~dp0start_daemon.bat\"" /sc onlogon /rl limited /f
if %errorlevel%==0 (
    echo [OK] 已注册计划任务 intero-daemon：下次登录 Windows 时自动启动。
    echo      立即启动一次：双击 scripts\start_daemon.bat
    echo      卸载：双击 scripts\uninstall_autostart.bat
) else (
    echo [FAIL] 注册失败，可能需要管理员权限或手动在任务计划程序中创建。
)
pause
