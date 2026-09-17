"""理睬反馈闭环 + 意图类型的黄金测试。"""

import json
import os
import tempfile
import time
import unittest

from intero.core import Intero
from intero.encoder import TfidfEncoder
from intero.intents import NullReminderExtractor, parse_reminders
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer


def make_organ(store_path: str) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "测试语料"])
    return Intero(
        encoder=enc,
        normalizer=NullNormalizer(),
        memory=TitansMemory(dim=128, hidden=256, depth=3, seed=0),
        store_path=store_path,
        extractor=NullReminderExtractor(),
    )


class TestIntentTypes(unittest.TestCase):
    def test_parse_four_types(self):
        resp = json.dumps([
            {"type": "remind", "payload": "交稿", "deadline": None, "urgency": 0.9},
            {"type": "followup", "payload": "追问装修进度", "urgency": 0.6},
            {"type": "care", "payload": "问问感冒好点没", "urgency": 0.7},
            {"type": "association", "payload": "新欢咖啡和旧偏好矛盾哦", "urgency": 0.5},
            {"type": "瞎编的", "payload": "非法类型回退", "urgency": 0.5},
        ])
        rems = parse_reminders(resp)
        self.assertEqual([r["type"] for r in rems],
                         ["remind", "followup", "care", "association", "remind"])


class TestFeedbackLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="intero_fb_")
        self.db = os.path.join(self.tmp, "content.db")

    def test_multiplier_prior_is_neutral(self):
        organ = make_organ(self.db)
        self.assertEqual(organ.urgency_multiplier("remind"), 1.0)

    def test_ignored_outreach_suppresses_future_urgency(self):
        """连续 3 次主动搭话无人理 → 同类意图紧迫度乘子降到 0.5。"""
        organ = make_organ(self.db)
        for _ in range(3):
            organ.record_delivery(["remind"], proactive=True)
            organ._beat()                                   # 用户交互，但标记已超窗？——未超窗会 ack
        # 上面 _beat 会把 10 分钟内的送达记为理睬；构造"无人理"需催老标记
        fb = organ._feedback()
        fb["remind"] = {"delivered": 3, "acked": 0}         # 直接造账：3 送 0 理
        organ._save_feedback(fb)
        self.assertEqual(organ.urgency_multiplier("remind"), 0.5)

    def test_ack_within_window(self):
        """主动送达后 10 分钟内用户来交互 → 记为理睬。"""
        organ = make_organ(self.db)
        organ.record_delivery(["care"], proactive=True)
        organ._beat()                                       # 用户来了（窗内）
        fb = organ._feedback()
        self.assertEqual(fb["care"], {"delivered": 1, "acked": 1})

    def test_no_ack_after_window(self):
        """标记超窗后不补记理睬。"""
        organ = make_organ(self.db)
        organ.record_delivery(["care"], proactive=True)
        raw = json.loads(organ.st.get_meta_text("last_proactive"))
        raw["ts"] -= 7200                                   # 催老 2 小时
        organ.st.set_meta_text("last_proactive", json.dumps(raw))
        organ._beat()
        fb = organ._feedback()
        self.assertEqual(fb["care"], {"delivered": 1, "acked": 0})

    def test_engaged_feedback_counts_all_acked(self):
        """对话内送达（recall）直接记为已理睬。"""
        organ = make_organ(self.db)
        organ.record_delivery(["followup"], proactive=False)
        fb = organ._feedback()
        self.assertEqual(fb["followup"], {"delivered": 1, "acked": 1})

    def test_add_intention_applies_multiplier(self):
        """注册意图时紧迫度被历史理睬率打折。"""
        organ = make_organ(self.db)
        organ._save_feedback({"remind": {"delivered": 4, "acked": 0}})
        organ.add_intention("remind", "又被无视的提醒", urgency=0.9, ttl=600)
        self.assertAlmostEqual(organ.hb.intentions[-1].urgency, 0.9 * 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
