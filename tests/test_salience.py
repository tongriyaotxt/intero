"""实体显著性门控的黄金测试（纵向模拟教训：写入率=召回天花板）。"""

import os
import tempfile
import unittest

from intero.core import Intero, is_salient
from intero.encoder import TfidfEncoder
from intero.intents import NullReminderExtractor
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer


def make_organ(store_path: str) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "攀岩", "量子", "语料"])
    return Intero(
        encoder=enc,
        normalizer=NullNormalizer(),
        memory=TitansMemory(dim=128, hidden=256, depth=3, seed=0),
        store_path=store_path,
        extractor=NullReminderExtractor(),
    )


class TestSalience(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="intero_sal_"), "c.db")

    def test_user_reference_regex(self):
        self.assertTrue(is_salient("用户喝咖啡不加糖"))
        self.assertTrue(is_salient("我下周二交稿"))
        self.assertFalse(is_salient("量子计算机取得突破"))

    def test_salient_bypasses_surprise_gate(self):
        """门控拒绝写入时，用户相关事实仍然写入；无关事实仍然被省。"""
        organ = make_organ(self.db)
        organ.mem.gates.should_write = lambda surprise: False   # 门控全关
        r1 = organ.ingest("用户每周三晚上去攀岩馆", normalize=False)
        r2 = organ.ingest("某国量子计算机取得突破", normalize=False)
        self.assertEqual(r1["written"], 1)      # 显著 → 绕门写入
        self.assertEqual(r2["written"], 0)      # 不显著不惊讶 → 省
        self.assertEqual(len(organ.st), 1)


if __name__ == "__main__":
    unittest.main()
