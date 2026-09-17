"""intents.py — 意图自生：从记忆事实里萌发提醒意图（不靠显式注册）

设计动机（tech-base/23 内态驱动目标生成的 sidecar 落地）：
  白天写入的事实里藏着时间承诺（"下周二下午3点交稿"、"周三去攀岩"）。
  显式 add_intention 是"你叫它记"，意图自生是"它自己想到"——
  dream 周期里用冻结 LLM 通读库存事实，抽出到期事项，注册成心跳意图。

红线（与 normalize.py 相同）：
  - LLM 只做文本理解，一个参数不动；
  - 无 LLM / 抽取失败 → 返回空，系统永不因此停摆；
  - 萌发可审计：每条意图 kind="sprout"，payload 人可读，来源事实 id 记录在 meta。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Protocol

from .normalize import _load_dotenv

PROMPT_VERSION = "v1"

PROMPT_TEMPLATE = """今天是 {today}。下面是关于用户的事实记忆。
请从中萌发值得主动提起的事项，分四类：
  remind      有明确时间、到期要提醒的事（约会/截止/计划/承诺）
  followup    看起来没聊完/没收尾、值得追问进展的话题
  care        涉及用户身心状态、值得稍后关心的事（生病/情绪/大事前后）
  association 新事实与旧事实之间的有趣联结，值得主动分享
规则：
1. 只输出 JSON 数组，每项：{{"type": "四类之一", "payload": "提醒/追问/关心/分享时要说的话（简短口语）", "deadline": "YYYY-MM-DDTHH:MM 或 null", "urgency": 0到1}};
2. 没有值得萌发的就输出 []；已经过期的事项跳过；不要解释。

事实记忆：
{facts}

输出："""

_ARRAY_RE = re.compile(r"\[.*\]", re.S)


INTENT_TYPES = {"remind", "followup", "care", "association"}


def parse_reminders(response: str) -> list[dict]:
    """从 LLM 响应抠 JSON 数组；每项规范化为 {type, payload, deadline_ts|None, urgency}。"""
    m = _ARRAY_RE.search(response)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for r in arr if isinstance(arr, list) else []:
        if not isinstance(r, dict) or not str(r.get("payload", "")).strip():
            continue
        dl_ts = None
        dl = r.get("deadline")
        if isinstance(dl, str) and dl.strip():
            try:
                dl_ts = datetime.fromisoformat(dl.strip()).timestamp()
            except ValueError:
                dl_ts = None
        try:
            urgency = float(r.get("urgency", 0.8))
        except (TypeError, ValueError):
            urgency = 0.8
        itype = str(r.get("type", "remind"))
        out.append({
            "type": itype if itype in INTENT_TYPES else "remind",
            "payload": str(r["payload"]).strip(),
            "deadline_ts": dl_ts,
            "urgency": max(0.0, min(1.0, urgency)),
        })
    return out


class ReminderExtractor(Protocol):
    name: str

    def extract(self, facts: list[str]) -> list[dict]:
        """事实列表 → 提醒事项列表。空列表 = 没什么值得提醒/抽取失败。"""


class NullReminderExtractor:
    """离线兜底：不萌发（等价于此功能关闭）。"""

    name = "null"

    def extract(self, facts: list[str]) -> list[dict]:
        return []


class LLMReminderExtractor:
    """OpenAI 兼容 API 抽取（配置来源与 normalizer 相同：.env / INTERO_LLM_*）。"""

    def __init__(self, base_url: str, api_key: str, model: str = "deepseek-chat") -> None:
        self.base_url = base_url
        self._key = api_key
        self.model = model
        self.name = f"llm:api:{model}"

    def _call(self, prompt: str) -> str:
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def extract(self, facts: list[str]) -> list[dict]:
        if not facts:
            return []
        prompt = PROMPT_TEMPLATE.format(
            today=datetime.now().strftime("%Y-%m-%d %H:%M"),
            facts="\n".join(f"- {f}" for f in facts),
        )
        try:
            return parse_reminders(self._call(prompt))
        except Exception:
            return []


def best_available_extractor(**kwargs) -> ReminderExtractor:
    """有 API 配置用 LLM；否则 Null（不萌发，系统照跑）。"""
    cfg = {**_load_dotenv(), **{k: v for k, v in os.environ.items() if k.startswith("INTERO_LLM_")}}
    base_url = kwargs.get("base_url") or cfg.get("INTERO_LLM_BASE_URL")
    api_key = kwargs.get("api_key") or cfg.get("INTERO_LLM_API_KEY")
    model = kwargs.get("model") or cfg.get("INTERO_LLM_MODEL", "deepseek-chat")
    if base_url and api_key and "填入" not in api_key:
        return LLMReminderExtractor(base_url=base_url, api_key=api_key, model=model)
    return NullReminderExtractor()
