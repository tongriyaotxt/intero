@echo off
rem uninstall_autostart.bat — 卸载 intero 开机自启
schtasks /delete /tn "intero-daemon" /f
if %errorlevel%==0 (echo [OK] 已卸载 intero-daemon 计划任务。) else (echo [提示] 任务不存在或已卸载。)
pause
