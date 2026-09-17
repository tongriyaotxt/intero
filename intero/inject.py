"""inject.py — prompt 注入装配：把检索结果排成可拼接的记忆块。

规则：
  - consolidated（晋升事实）排最前，conflict 带标注（供 LLM 裁决，不替它做决定）；
  - dedup / noise 不上墙；
  - 字符预算内截断，超预算宁可少给不给残句；
  - 空结果返回空串（调用方拼 prompt 时无记忆就什么都不加）。
"""

from __future__ import annotations

HEADER = "【长期记忆】以下是与当前对话相关的用户事实，仅供参考，矛盾处已标注："
CONFLICT_MARK = "⚠️ 疑似矛盾"


def assemble(results: list[dict], topk: int = 5, budget_chars: int = 600) -> str:
    if not results:
        return ""
    priority = {"consolidated": 0, "fact": 1, "composite": 1, "conflict": 2}
    rows = [r for r in results if r.get("kind") not in ("dedup", "noise")]
    rows.sort(key=lambda r: (priority.get(r.get("kind"), 1), -r["score"]))
    lines: list[str] = []
    used = len(HEADER)
    for r in rows[:topk]:
        text = r["text"]
        if r.get("kind") == "conflict":
            text = f"{CONFLICT_MARK}：{text}"
        cost = len(text) + 2
        if used + cost > budget_chars:
            break
        lines.append(f"- {text}")
        used += cost
    if not lines:
        return ""
    return HEADER + "\n" + "\n".join(lines)
