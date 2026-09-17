"""client.py — intero 常驻服务的瘦客户端（MCP server 转发壳用）。

服务端没起时 is_alive() 快速失败（0.4s 超时），调用方回退本地 Intero——
服务是加速层，不是单点依赖。
"""

from __future__ import annotations

import json
import os
import urllib.request

from .service import DEFAULT_PORT


class RemoteOrgan:
    """与 Intero 同形的远端门面（只覆盖 MCP 工具用到的方法子集）。"""

    def __init__(self, base: str | None = None, timeout: float = 30.0) -> None:
        self.base = (base or os.environ.get("INTERO_SERVER")
                     or f"http://127.0.0.1:{DEFAULT_PORT}").rstrip("/")
        self.timeout = timeout
        self.name = f"remote:{self.base}"

    def _post(self, path: str, body: dict, timeout: float | None = None):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            out = json.load(r)
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def is_alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=0.4) as r:
                return json.load(r).get("ok", False)
        except Exception:
            return False

    # ---- 与 Intero 同形的方法子集 ----

    def ingest(self, text: str, kind: str = "fact") -> dict:
        return self._post("/write", {"text": text, "kind": kind})

    def recall(self, query: str, topk: int = 5, budget_chars: int = 600) -> str:
        return self._post("/recall", {"query": query, "topk": topk})["block"]

    def status(self) -> dict:
        return self._post("/status", {})

    def add_intention(self, kind: str, payload: str, urgency: float = 0.5, ttl: float = 3600.0) -> dict:
        return self._post("/add_intention",
                          {"kind": kind, "payload": payload, "urgency": urgency, "ttl": ttl})

    def tick(self) -> dict:
        return self._post("/tick", {})

    def dream_cycle(self) -> dict:
        return self._post("/dream", {})
