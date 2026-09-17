"""normalize.py — 写入前事实归一化（canonicalization）。

设计动机（2026-09-16 架构决策，替代原 M1.5 微调主线）：
  开源 encoder 全员败于整句否定（"我从来不喝咖啡只喝茶"与"我喝咖啡不加糖"
  余弦 0.78，比同义改写还高，见 bench/CALIBRATION.md Step 0 实录）。
  与其微调 encoder 修一个坑，不如让系统里本来就有的冻结 LLM 把用户原话
  先改写成**原子事实**——否定、指代、复合句一类语言学陷阱整体绕过，
  且归一化结果人可读、可审计。encoder 微调降级为无 LLM 环境的离线兜底档。

红线：
  - LLM 只做文本改写，一个参数不动（sidecar 原则不破）；
  - 归一化失败/无 LLM 时原样透传（NullNormalizer 语义），系统永不因此停摆；
  - 结果带磁盘缓存：同一原话只付一次 LLM 调用。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol

PROMPT_VERSION = "v2"  # 改 prompt 时递增，让旧缓存自动失效

PROMPT_TEMPLATE = """把下面这句话改写成原子事实列表。规则：
1. 每条事实必须自含完整语义：限定词（不加糖/双份/从来不）不许和它的对象拆开；
2. 否定语义必须显式保留在所属事实里（不/没有/从来不）；
3. 只有真正独立的子句才拆成多条；宁可一条完整事实，不要两条残缺事实；
4. 指代能确定就替换成实体；只输出 JSON 数组，不要解释。

示例：
原话：我喝咖啡从来不加糖
输出：["用户喝咖啡从来不加糖"]
原话：我从来不喝咖啡，只喝茶
输出：["用户从来不喝咖啡", "用户只喝茶"]
原话：我喝咖啡要加双份糖
输出：["用户喝咖啡要加双份糖"]

原话：{text}
输出："""

_FACTS_RE = re.compile(r"\[.*\]", re.S)


def parse_facts(response: str) -> list[str]:
    """从 LLM 响应里抠出 JSON 数组；失败返回空列表（调用方决定回退）。"""
    m = _FACTS_RE.search(response)
    if not m:
        return []
    try:
        facts = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]


class Normalizer(Protocol):
    name: str

    def normalize(self, text: str) -> list[str]:
        """原话 → 原子事实列表。空列表 = 失败，调用方回退原文。"""


class NullNormalizer:
    """离线兜底：原样透传（等价于归一化前置不存在）。"""

    name = "null"

    def normalize(self, text: str) -> list[str]:
        return [text]


class LLMNormalizer:
    """LLM 归一化。后端二选一：
      - "api"：OpenAI 兼容 chat/completions（base_url/api_key/model 显式给）
      - "local"：本地 HF 模型（默认缓存里的 Qwen/Qwen2-VL-2B-Instruct，CPU 可跑）
    """

    def __init__(
        self,
        backend: str = "local",
        model: str = "Qwen/Qwen2-VL-2B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        cache_path: str | os.PathLike = ".normalize_cache.json",
        max_new_tokens: int = 128,
    ) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self._key = api_key
        self.max_new_tokens = max_new_tokens
        self.name = f"llm:{backend}:{model}"
        self._cache_path = Path(cache_path)
        self._cache: dict[str, list[str]] = {}
        if self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
        self._lm = None  # 本地模型懒加载

    # ---------------- 后端调用 ----------------

    def _call_api(self, prompt: str) -> str:
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
            data = json.load(r)
        return data["choices"][0]["message"]["content"]

    def _call_local(self, prompt: str) -> str:
        if self._lm is None:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model)
            self._lm = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model, torch_dtype=torch.float32
            )
            self._lm.eval()
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], return_tensors="pt")
        import torch

        with torch.no_grad():
            out = self._lm.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode([gen], skip_special_tokens=True)[0]

    # ---------------- 主接口 ----------------

    def normalize(self, text: str) -> list[str]:
        key = hashlib.blake2b(f"{PROMPT_VERSION}|{self.name}|{text}".encode(), digest_size=16).hexdigest()
        if key in self._cache:
            return self._cache[key]
        prompt = PROMPT_TEMPLATE.format(text=text)
        try:
            raw = self._call_api(prompt) if self.backend == "api" else self._call_local(prompt)
            facts = parse_facts(raw)
        except Exception:
            facts = []
        if facts:
            self._cache[key] = facts
            self._cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        return facts


def _load_dotenv(path: str | os.PathLike = ".env") -> dict[str, str]:
    """极简 .env 解析（KEY=VALUE，# 注释），不引依赖。"""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()  # 剥行内注释
            out[k.strip()] = v
    return out


def best_available_normalizer(**kwargs) -> Normalizer:
    """优先 .env/环境变量里的 API 配置；无配置则本地模型；都不可用则透传。"""
    cfg = {**_load_dotenv(), **{k: v for k, v in os.environ.items() if k.startswith("INTERO_LLM_")}}
    base_url = kwargs.get("base_url") or cfg.get("INTERO_LLM_BASE_URL")
    api_key = kwargs.get("api_key") or cfg.get("INTERO_LLM_API_KEY")
    model = kwargs.get("model") or cfg.get("INTERO_LLM_MODEL", "deepseek-chat")
    if base_url and api_key and "填入" not in api_key:
        return LLMNormalizer(
            backend="api", model=model, base_url=base_url, api_key=api_key,
            cache_path=kwargs.get("cache_path", ".normalize_cache.json"),
        )
    try:
        return LLMNormalizer(backend="local", cache_path=kwargs.get("cache_path", ".normalize_cache.json"))
    except Exception:
        return NullNormalizer()
