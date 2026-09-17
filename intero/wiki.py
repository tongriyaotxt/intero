"""wiki.py — 只读导出层：sqlite 内容库 → 人可读 Markdown wiki（派生物，非事实源）

定位（用户拍板 2026-09-17）：sqlite 仍是唯一事实源；wiki 是每晚 dream 后
重新生成的**只读视图**——能看、能审计，编辑不生效（下次导出即覆盖，文件头有声明）。
主动性可见化：心跳状态页导出意图队列/待说事项/理睬账本——它"在想什么"对人透明。

布局：
  README.md      索引 + 只读声明
  用户画像.md     consolidated 事实（dream 晋升，"记得牢的"）
  事实库.md       普通 fact/composite（按类别从文本启发式归组）
  矛盾与仲裁.md   conflict 对（读给人裁决，同 LLM 注入块的语义）
  日志/YYYY-MM-DD.md  按写入日期分组（全部条目，含归档 dedup 的标注）
  心跳状态.md     意图队列 / 待说事项 / 理睬账本 / λ / 库存
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

READONLY_BANNER = (
    "> ⚠️ **只读视图，请勿编辑**——本文件由 intero 导出层自动生成，"
    "每次 dream/手动导出时全量覆盖。修改请通过对话进行（对它说『忘掉/记住…』）。\n\n"
)

CAT_RULES = [("偏好", ("喜欢", "偏好", "习惯", "从来不", "爱", "常")),
             ("计划", ("要", "计划", "打算", "截止", "前", "明天", "下周", "周")),
             ("社交", ("朋友", "同事", "室友", "家人", "妈", "爸", "哥", "姐")),
             ("工作", ("工作", "项目", "上线", "代码", "会议", "老板"))]


def _category(text: str) -> str:
    for name, kws in CAT_RULES:
        if any(k in text for k in kws):
            return name
    return "其他"


def _page(title: str, body: str) -> str:
    return f"# {title}\n\n{READONLY_BANNER}{body}"


def render(store, hb_snapshot: dict | None, feedback: dict,
           recent_said: list[dict] | None = None) -> dict[str, str]:
    """生成全部页面 {相对路径: 内容}。store=ContentStore, hb_snapshot=心跳快照 dict。"""
    items = store.items()                      # 存活条目（含 kind/vec）
    pages: dict[str, str] = {}

    # —— 用户画像（晋升层） ——
    cons = [it for it in items if it["kind"] == "consolidated"]
    body = "dream 晋升的\"记得牢的\"事实（按类别归组）：\n\n" if cons else "（还空着——跑几次 dream 后，记得牢的事实会晋升到这里）\n"
    by_cat: dict[str, list[str]] = {}
    for it in cons:
        by_cat.setdefault(_category(it["text"]), []).append(it["text"])
    for cat, texts in sorted(by_cat.items()):
        body += f"## {cat}\n" + "".join(f"- {t}\n" for t in texts) + "\n"
    pages["用户画像.md"] = _page("用户画像", body)

    # —— 事实库（未晋升层） ——
    facts = [it for it in items if it["kind"] in ("fact", "composite")]
    body = "尚未晋升的普通事实：\n\n" if facts else "（空）\n"
    by_cat = {}
    for it in facts:
        by_cat.setdefault(_category(it["text"]), []).append(it["text"])
    for cat, texts in sorted(by_cat.items()):
        body += f"## {cat}\n" + "".join(f"- {t}\n" for t in texts) + "\n"
    pages["事实库.md"] = _page("事实库", body)

    # —— 矛盾与仲裁 ——
    conf = [it for it in items if it["kind"] == "conflict"]
    body = ("疑似矛盾的事实对（都可能是真的——不同时期/不同语境）。"
            "对它说\"以 X 为准\"即可仲裁：\n\n" + "".join(f"- ⚠️ {it['text']}\n" for it in conf)
            ) if conf else "（当前没有标记中的矛盾）\n"
    pages["矛盾与仲裁.md"] = _page("矛盾与仲裁", body)

    # —— 日志（按日分组，含归档标注） ——
    by_day: dict[str, list[str]] = {}
    for it in items:
        day = datetime.fromtimestamp(it.get("ts", 0)).strftime("%Y-%m-%d") if it.get("ts") else "未知日期"
        mark = {"dedup": "（已归档·重复）", "conflict": "⚠️ "}.get(it["kind"], "")
        by_day.setdefault(day, []).append(f"- {mark}{it['text']}")
    for day, lines in sorted(by_day.items()):
        pages[f"日志/{day}.md"] = _page(f"日志 {day}", "\n".join(lines) + "\n")

    # —— 心跳状态（主动性可见化） ——
    snap = hb_snapshot or {}
    body = f"- 状态：**{snap.get('state', '?')}**\n- 库存：{len(store)} 条　λ：{store.lam:.3f}\n\n"
    ints = snap.get("intentions", [])
    body += "## 意图队列（它正在\"想\"的事）\n\n" + (
        "".join(f"- [{i['kind']}] {i['payload']}（紧迫度 {i['urgency']}）\n" for i in ints)
        if ints else "（空——它现在没什么想说的）\n")
    pend = snap.get("pending", [])
    body += "\n## 待说事项（已赢拍卖、等你见面）\n\n" + (
        "".join(f"- [{p['kind']}] {p['payload']}\n" for p in pend) if pend else "（空）\n")
    said = recent_said or []
    body += "\n## 它曾主动说（对话线头，可以接着唠）\n\n" + (
        "".join(f"- {datetime.fromtimestamp(s['ts']).strftime('%m-%d %H:%M')} {s['text']}\n"
                for s in said[-10:])
        if said else "（它还没主动说过话）\n")
    body += "\n## 理睬账本（它学到的\"说什么你会理\"）\n\n" + (
        "".join(f"- {k}：送达 {v['delivered']} 次 / 被理睬 {v['acked']} 次\n"
                for k, v in sorted(feedback.items()))
        if feedback else "（还没有送达记录）\n")
    pages["心跳状态.md"] = _page("心跳状态", body)

    # —— 索引 ——
    idx = ("intero 记忆的只读 wiki 视图（每次 dream 后重新生成）。\n\n"
           "- [[用户画像]] —— 它记得牢的你\n"
           "- [[事实库]] —— 还没晋升的普通记忆\n"
           "- [[矛盾与仲裁]] —— 需要你裁决的矛盾\n"
           "- [[心跳状态]] —— 它在想什么、打算说什么\n"
           "- 日志/ —— 全部记忆按天\n")
    pages["README.md"] = _page("intero 记忆 wiki", idx)
    return pages


def export_wiki(store, hb_snapshot: dict | None, feedback: dict,
                out_dir: str | Path = ".intero/wiki",
                recent_said: list[dict] | None = None) -> int:
    """全量重写导出目录。返回写入文件数。"""
    out = Path(out_dir)
    pages = render(store, hb_snapshot, feedback, recent_said)
    if out.exists():                       # 全量覆盖前清旧（日志页可能减少）
        for p in out.rglob("*.md"):
            p.unlink()
    for rel, content in pages.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return len(pages)
