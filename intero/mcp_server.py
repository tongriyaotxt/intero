"""mcp_server.py — M4 接入：intero 作为 MCP server 暴露给任意 LLM 宿主

工具：
  memory_write(text)   写入（归一化 + 门控）
  memory_recall(query) 检索并装配 prompt 注入块
  memory_status()      器官状态（写入/库存/λ/回滚）

运行：PYTHONPATH=. python -m intero.mcp_server   （stdio 传输，Claude/Kimi 等宿主可直接挂）
依赖：pip install mcp（pyproject 已声明 optional extra "mcp"）
"""

from __future__ import annotations

from .core import Intero

_INTERO: Intero | None = None


def _organ() -> Intero:
    global _INTERO
    if _INTERO is None:
        _INTERO = Intero()
    return _INTERO


def memory_write(text: str, kind: str = "fact") -> dict:
    """把一条用户事实写入长期记忆（先 LLM 归一化为原子事实，再过惊讶门控）。"""
    return _organ().ingest(text, kind=kind)


def memory_recall(query: str, topk: int = 5) -> str:
    """检索与 query 相关的记忆，返回可直接拼进 prompt 的记忆块（无记忆则空串）。"""
    return _organ().recall(query, topk=topk)


def memory_status() -> dict:
    """记忆器官状态：摄入/写入/库存/λ/回滚/encoder/normalizer。"""
    return _organ().status()


def main() -> None:
    try:   # mcp 2.x：FastMCP 改名 MCPServer
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:   # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    mcp = Server("intero-memory")
    mcp.tool()(memory_write)
    mcp.tool()(memory_recall)
    mcp.tool()(memory_status)
    mcp.run()


if __name__ == "__main__":
    main()
