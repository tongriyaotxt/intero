"""对话线头的黄金测试：daemon 主动说过的话 → 下次 recall 顶部可见 → 能接着唠。"""

import json
import os
import tempfile
import time
import unittest

from intero.core import Intero
from intero.encoder import TfidfEncoder
from intero.intents import NullReminderExtractor
from intero.normalize import NullNormalizer


def make_organ(store_path: str) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "攀岩", "测试语料"])
    return Intero(encoder=enc, normalizer=NullNormalizer(), memory=None, mode="light",
                  store_path=store_path, extractor=NullReminderExtractor())


class TestConversationThread(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="intero_thread_"), "c.db")

    def test_proactive_said_appears_in_recall(self):
        """daemon 弹窗送达（带润色文本）→ 下次 recall 顶部【它曾主动说】可见。"""
        organ = make_organ(self.db)
        organ.record_delivery(["care"], proactive=True,
                              texts=["主人，两天没见啦，最近在忙啥？"])
        block = organ.recall("随便聊聊")
        self.assertIn("【它曾主动说】", block)
        self.assertIn("两天没见啦", block)

    def test_said_expires_after_48h(self):
        """48 小时前的主动发言不再进注入块（线头也会过期）。"""
        organ = make_organ(self.db)
        organ.record_delivery(["care"], proactive=True, texts=["很久以前的话"])
        said = json.loads(organ.st.get_meta_text("recent_said"))
        said[0]["ts"] -= 49 * 3600
        organ.st.set_meta_text("recent_said", json.dumps(said))
        self.assertNotIn("【它曾主动说】", organ.recall("聊聊"))

    def test_said_cap_20(self):
        organ = make_organ(self.db)
        for i in range(25):
            organ.record_delivery(["remind"], proactive=True, texts=[f"第{i}句"])
        said = json.loads(organ.st.get_meta_text("recent_said"))
        self.assertEqual(len(said), 20)
        self.assertEqual(said[-1]["text"], "第24句")

    def test_said_persists_across_respawn(self):
        """线头跨进程存活（spawn-per-call 纪律）。"""
        a = make_organ(self.db)
        a.record_delivery(["remind"], proactive=True, texts=["跨进程线头"])
        a.st.close()
        b = make_organ(self.db)
        self.assertIn("跨进程线头", b.recall("接着聊"))
        b.st.close()

    def test_wiki_shows_said(self):
        organ = make_organ(self.db)
        organ.record_delivery(["remind"], proactive=True, texts=["wiki 里也该看到这句"])
        tmp = os.path.dirname(self.db)
        os.chdir(tmp)
        organ.export_wiki()
        hb = open(os.path.join(tmp, ".intero", "wiki", "心跳状态.md"), encoding="utf-8").read()
        self.assertIn("wiki 里也该看到这句", hb)


if __name__ == "__main__":
    unittest.main()
