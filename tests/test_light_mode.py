"""light 模式的黄金测试（2026-09-17 架构决策：Titans 降级实验层，light 为默认）。

light 门控 = 显著 ∨ 新颖（冗余度 < novelty_cap）；dream = 策展 + 重复巩固晋升；
召回行为与 full 相同（λ 实证全程 0，读出本来就是纯向量）。
"""

import os
import tempfile
import unittest

import numpy as np

from intero.core import Intero
from intero.encoder import TfidfEncoder
from intero.intents import NullReminderExtractor
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer


def make_light(store_path: str, **kw) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "攀岩", "量子", "突破口", "语料"])
    return Intero(
        encoder=enc, normalizer=NullNormalizer(), memory=None, mode="light",
        store_path=store_path, extractor=NullReminderExtractor(), **kw)


class TestLightMode(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="intero_light_"), "c.db")

    def test_default_is_light(self):
        """不显式传 memory 时默认 light（用户拍板的新默认）。"""
        organ = Intero(encoder=__import__("intero.encoder", fromlist=["TfidfEncoder"]).TfidfEncoder(dim=128),
                       normalizer=NullNormalizer(), extractor=NullReminderExtractor(),
                       store_path=self.db)
        self.assertEqual(organ.mode, "light")
        self.assertIsNone(organ.mem)

    def test_passing_memory_is_full(self):
        organ = Intero(encoder=None or __import__("intero.encoder", fromlist=["TfidfEncoder"]).TfidfEncoder(dim=128),
                       normalizer=NullNormalizer(), extractor=NullReminderExtractor(),
                       memory=TitansMemory(dim=128, hidden=64, depth=2, seed=0),
                       store_path=self.db)
        self.assertEqual(organ.mode, "full")

    def test_gate_salient_or_novel(self):
        organ = make_light(self.db)
        self.assertEqual(organ.ingest("用户每周三晚上去攀岩馆", normalize=False)["written"], 1)   # 显著
        self.assertEqual(organ.ingest("某国量子计算机取得重大突破", normalize=False)["written"], 1)  # 新颖
        dup = organ.ingest("某国量子计算机取得重大突破", normalize=False)                       # 逐字重复
        self.assertEqual(dup["written"], 0)                                                  # 冗余 ≥ cap → 拦

    def test_recall_and_dream_light(self):
        organ = make_light(self.db)
        organ.ingest("用户喝咖啡从来不加糖", normalize=False)
        organ.ingest("用户喝咖啡从不加糖的习惯保持多年", normalize=False)   # 近重复 → 熟悉度晋升材料
        block = organ.recall("我喝咖啡加不加糖")
        self.assertIn("不加糖", block) or self.assertIn("不加糖", block.replace("不", "不"))
        os.chdir(os.path.dirname(self.db))
        r = organ.dream_cycle()
        self.assertIn("晋升", r)
        self.assertNotIn("自重构误差", r)         # light 无 Titans 重构误差
        st = organ.status()
        self.assertEqual(st["模式"], "light")
        self.assertEqual(st["写入"], "-")


if __name__ == "__main__":
    unittest.main()
