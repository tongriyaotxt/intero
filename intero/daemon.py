"""daemon.py — intero 守护进程：主动搭话的节拍源（独立于宿主常驻）

与 MCP server 的分工：
  MCP server  被动应答——宿主（kimi）调用才活，记忆读写走这里；
  daemon      常驻自持——按心率跳拍，拍卖赢出的冲动**主动送达**给用户本人。

送达路由（一行说清）：
  你在聊（ENGAGED）→ 冲动留在 pending，等对话内 recall 送达（不打扰）；
  你不在（WATCH/DREAM）→ Windows 气泡通知 + 可选语音（--voice）主动搭话。

并发纪律：daemon 与 MCP 进程共享同一 sqlite。daemon 每拍前先 reload 快照
（吸收 MCP 侧新写入的意图/交互时间），跳完即存——进程间只通过库通信。

运行：
  INTERO_STORE=.intero/content.db HF_HUB_OFFLINE=1 PYTHONPATH=. \\
      python -m intero.daemon [--voice] [--once]
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import time

from .core import Intero
from .heartbeat import HeartState, Intention
from .service import DEFAULT_PORT, serve_in_thread


# ---- 通知通道（Windows 原生，零第三方依赖） ----

def _ps(script: str) -> None:
    """执行 PowerShell 脚本。UTF-16LE base64 传递，绕开 GBK 控制台乱码。"""
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    subprocess.run(
        ["powershell", "-NoProfile", "-EncodedCommand", enc],
        capture_output=True, check=False,
    )


def _q(text: str) -> str:
    """PowerShell 单引号字符串转义。"""
    return "'" + text.replace("'", "''") + "'"


def balloon(title: str, message: str, seconds: int = 8) -> None:
    """Windows 托盘气泡通知。"""
    _ps(f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.Visible = $true
$n.ShowBalloonTip({seconds * 1000}, {_q(title)}, {_q(message)}, [System.Windows.Forms.ToolTipIcon]::Info)
Start-Sleep -Seconds {seconds + 1}
$n.Dispose()
""")


def speak(text: str) -> None:
    """SAPI 语音播报（系统自带中文语音则直接说中文）。"""
    _ps(f"""
$v = New-Object -ComObject SAPI.SpVoice
$v.Rate = 0
[void]$v.Speak({_q(text)})
""")


def notify(intention: Intention, voice: bool = False) -> None:
    """默认通道：气泡 + （可选）语音。"""
    balloon("intero 主动搭话", intention.payload)
    if voice:
        speak(intention.payload)


# ---- 主循环 ----

def deliver_due(organ: Intero, notify_fn=notify, voice: bool = False) -> int:
    """把"该主动说的"送出去。返回送达条数。

    ENGAGED（用户正在对话）→ 不送，留给对话内 recall；
    WATCH/DREAM → 立即主动送达并清除。
    """
    if organ.hb.state == HeartState.ENGAGED:
        return 0
    pending = organ.hb.deliver_pending()
    for p in pending:
        notify_fn(p, voice=voice)
    if pending:
        organ.save_heartbeat()
        organ.record_delivery([p.kind for p in pending], proactive=True)
    return len(pending)


def run(organ: Intero, voice: bool = False, once: bool = False,
        notify_fn=notify, out=sys.stdout) -> None:
    print(f"[intero-daemon] 起搏开始（心率随状态变：1s/10s/60s）voice={voice}", file=out, flush=True)
    prev_state = None                       # None → 首拍若在 DREAM 也会跑一次夜间周期
    while True:
        organ.reload_heartbeat()          # 吸收 MCP 侧的新意图/交互时间
        r = organ.daemon_tick()           # 只跳拍，不记交互（daemon 不是用户）
        n = deliver_due(organ, notify_fn=notify_fn, voice=voice)
        print(f"[intero-daemon] tick 状态={r['状态']} 意图={r['意图队列']} "
              f"待说={r['待说']} 本次主动送达={n}", file=out, flush=True)
        # 进入 DREAM（休眠）的第一拍：跑夜间周期——回放巩固 + 意图自生
        if organ.hb.state == HeartState.DREAM and prev_state != HeartState.DREAM:
            report = organ.dream_cycle()
            print(f"[intero-daemon] 夜间周期: {report}", file=out, flush=True)
        prev_state = organ.hb.state
        if once:
            return
        time.sleep(max(1.0, organ.hb.rate))


def main() -> None:
    ap = argparse.ArgumentParser(description="intero 主动搭话守护进程")
    ap.add_argument("--voice", action="store_true", help="气泡之外再语音播报")
    ap.add_argument("--once", action="store_true", help="只跳一拍（调试用）")
    ap.add_argument("--serve", action="store_true", help="内嵌常驻 HTTP 服务（MCP 走它，毫秒级）")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    organ = Intero()
    if args.serve:
        serve_in_thread(organ, port=args.port)
        print(f"[intero-daemon] 常驻服务已内嵌于 :{args.port}", flush=True)
    run(organ, voice=args.voice, once=args.once)


if __name__ == "__main__":
    main()
