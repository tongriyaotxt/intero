"""wiki 导出层的黄金测试：只读声明 / 页面结构 / 心跳可见化 / dream 自动导出。"""

import os
import tempfile
import unittest

from intero.core import Intero
from intero.encoder import TfidfEncoder
from intero.intents import NullReminderExtractor
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer
from intero.wiki import READONLY_BANNER, export_wiki


def make_organ(store_path: str) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "攀岩", "交稿", "测试语料"])
    return Intero(
        encoder=enc,
        normalizer=NullNormalizer(),
        memory=TitansMemory(dim=128, hidden=256, depth=3, seed=0),
        store_path=store_path,
        extractor=NullReminderExtractor(),
    )


class TestWikiExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="intero_wiki_")
        self.db = os.path.join(self.tmp, "content.db")
        self.dir = os.path.join(self.tmp, "wiki")

    def test_pages_structure_and_readonly(self):
        organ = make_organ(self.db)
        organ.ingest("用户喝咖啡从来不加糖", normalize=False)
        organ.ingest("用户下周二下午3点要交知乎初稿", normalize=False)
        organ.add_intention("remind", "测试意图", urgency=0.9, ttl=600)
        organ.record_delivery(["remind"], proactive=False)
        n = export_wiki(organ.st, organ.hb.snapshot(), organ._feedback(), self.dir)
        self.assertGreaterEqual(n, 5)                       # README+画像+事实库+矛盾+心跳+日志
        # 每页都有只读声明
        for p in __import__("pathlib").Path(self.dir).rglob("*.md"):
            self.assertIn("只读视图", p.read_text(encoding="utf-8"))
        # 事实出现在日志和事实库
        facts = open(os.path.join(self.dir, "事实库.md"), encoding="utf-8").read()
        self.assertIn("喝咖啡", facts)
        # 心跳状态页：意图与理睬账本可见
        hb = open(os.path.join(self.dir, "心跳状态.md"), encoding="utf-8").read()
        self.assertIn("测试意图", hb)
        self.assertIn("送达 1 次", hb)

    def test_dream_cycle_auto_exports(self):
        """dream_cycle 结束自动刷新 wiki（返回报告含 wiki 页数）。"""
        organ = make_organ(self.db)
        organ.ingest("用户每周三晚上去攀岩馆", normalize=False)
        os.chdir(self.tmp)                                  # export 默认相对路径 .intero/wiki
        r = organ.dream_cycle()
        self.assertIn("wiki", r)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, ".intero", "wiki", "README.md")))

    def test_readonly_banner_text(self):
        self.assertIn("请勿编辑", READONLY_BANNER)


if __name__ == "__main__":
    unittest.main()
