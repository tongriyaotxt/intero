"""mcp_server.py — M4 接入：intero 作为 MCP server 暴露给任意 LLM 宿主

工具：
  memory_write(text)      写入（归一化 + 门控）
  memory_recall(query)    检索并装配 prompt 注入块（待说事项置顶送达）
  memory_status()         器官状态（写入/库存/λ/回滚/心跳）
  add_intention(...)      注册一条主动意图（冲动排队，见面先说）
  heartbeat_tick()        显式跳一拍（宿主每轮对话结束可调）

运行：PYTHONPATH=. python -m intero.mcp_server   （stdio 传输，Claude/Kimi 等宿主可直接挂）
依赖：pip install mcp（pyproject 已声明 optional extra "mcp"）
"""

from __future__ import annotations

from .core import Intero

_INTERO = None


def _organ():
    """优先连常驻服务（毫秒级）；服务没起则回退本地 Intero（慢但自治）。"""
    global _INTERO
    if _INTERO is None:
        from .client import RemoteOrgan

        remote = RemoteOrgan()
        _INTERO = remote if remote.is_alive() else Intero()
    return _INTERO


def memory_write(text: str, kind: str = "fact") -> dict:
    """把一条用户事实写入长期记忆（先 LLM 归一化为原子事实，再过惊讶门控）。"""
    return _organ().ingest(text, kind=kind)


def memory_recall(query: str, topk: int = 5) -> str:
    """检索与 query 相关的记忆，返回可直接拼进 prompt 的记忆块（无记忆则空串）。"""
    return _organ().recall(query, topk=topk)


def memory_status() -> dict:
    """记忆器官状态：摄入/写入/库存/λ/回滚/encoder/normalizer/心跳。"""
    return _organ().status()


def add_intention(kind: str, payload: str, urgency: float = 0.5, ttl: float = 3600.0) -> dict:
    """注册一条主动意图（如提醒/追问）。心跳拍卖裁决时机，赢出的冲动会在下次 recall 时置顶送达。

    kind: 意图类型（remind/followup/...）；payload: 要说的事（提示，不是最终话术）；
    urgency: 紧迫度 0~1（>0.55 才有机会压过沉默底价）；ttl: 时效秒数，过期作废。
    """
    return _organ().add_intention(kind=kind, payload=payload, urgency=urgency, ttl=ttl)


def heartbeat_tick() -> dict:
    """让心跳跳一拍：清理过期意图 + 主动性拍卖。宿主每轮对话结束时调用一次即可。"""
    return _organ().tick()


def dream_now() -> dict:
    """立即跑一个夜间周期：dream 回放/策展/晋升 + 意图自生（从记忆事实里萌发提醒意图）。"""
    return _organ().dream_cycle()


def main() -> None:
    try:   # mcp 2.x：FastMCP 改名 MCPServer
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:   # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    mcp = Server("intero-memory")
    mcp.tool()(memory_write)
    mcp.tool()(memory_recall)
    mcp.tool()(memory_status)
    mcp.tool()(add_intention)
    mcp.tool()(heartbeat_tick)
    mcp.tool()(dream_now)
    mcp.run()


if __name__ == "__main__":
    main()
