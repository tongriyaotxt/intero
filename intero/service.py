"""service.py — intero 常驻 HTTP 服务：bge 常驻内存，调用毫秒级（杀 spawn 冷启动）

动机（实测教训）：kimi spawn-per-call 架构下每个 MCP 工具调用都新建进程 +
重载 encoder（~15s），真实对话不可用。服务化后：
  - daemon（或服务单独跑）持有常驻 Intero，encoder 只加载一次；
  - MCP server 瘦成转发壳（client.py），单次调用 = 一次 localhost HTTP（毫秒）。

并发纪律：一把大锁包住 organ 全部操作（个人记忆系统 QPS≈0，
简单正确 > 精巧并发）。HTTP 层只绑 127.0.0.1——记忆是隐私数据，永不监听外部。

端点（全部 POST JSON，除 /health 可 GET）：
  /write        {text, kind?}              → ingest
  /recall       {query, topk?}             → 注入块文本
  /status       {}                         → status
  /add_intention{kind, payload, urgency?, ttl?}
  /tick         {}                         → 心跳拍卖结果
  /dream        {}                         → 夜间周期报告
  /health       → {"ok": true, "uptime": s}
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import Intero

DEFAULT_PORT = 7377


class _Handler(BaseHTTPRequestHandler):
    organ: Intero = None           # 由 serve() 注入
    lock: threading.Lock = None
    started: float = 0.0

    def log_message(self, *a):     # 静音 access log
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "uptime": round(time.time() - self.started, 1)})
        else:
            self._json(404, {"error": "unknown path"})

    def do_POST(self):
        h, lock, organ = self, self.lock, self.organ
        b = h._body()
        try:
            with lock:
                if h.path == "/write":
                    out = organ.ingest(b.get("text", ""), kind=b.get("kind", "fact"))
                elif h.path == "/recall":
                    out = {"block": organ.recall(b.get("query", ""), topk=int(b.get("topk", 5)))}
                elif h.path == "/status":
                    out = organ.status()
                elif h.path == "/add_intention":
                    out = organ.add_intention(
                        b.get("kind", "remind"), b.get("payload", ""),
                        urgency=float(b.get("urgency", 0.5)), ttl=float(b.get("ttl", 3600)))
                elif h.path == "/tick":
                    out = organ.tick()
                elif h.path == "/dream":
                    out = organ.dream_cycle()
                else:
                    self._json(404, {"error": "unknown path"})
                    return
            self._json(200, out)
        except Exception as e:                       # 服务永不因单请求崩
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def make_server(organ: Intero, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    _Handler.organ = organ
    _Handler.lock = threading.Lock()
    _Handler.started = time.time()
    return ThreadingHTTPServer(("127.0.0.1", port), _Handler)


def serve(organ: Intero, port: int = DEFAULT_PORT) -> None:
    """前台跑服务（阻塞）。"""
    srv = make_server(organ, port)
    print(f"[intero-service]  listening on http://127.0.0.1:{port} （仅本机）", flush=True)
    srv.serve_forever()


def serve_in_thread(organ: Intero, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """后台线程跑服务（daemon 内嵌用）。返回 server 便于关停。"""
    srv = make_server(organ, port)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="intero-service")
    t.start()
    return srv
