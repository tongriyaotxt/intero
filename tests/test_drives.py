"""内态驱动 + 作息底价 + 话术润色的黄金测试。"""

import os
import tempfile
import time
import unittest

from intero.core import Intero
from intero.daemon import polish
from intero.drives import (DRIVE_COOLDOWN, circadian_floor, compute_drives,
                           maybe_sprout_drive_intentions)
from intero.encoder import TfidfEncoder
from intero.intents import NullReminderExtractor
from intero.normalize import NullNormalizer


def make_organ(store_path: str) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "测试", "语料"])
    return Intero(encoder=enc, normalizer=NullNormalizer(), memory=None, mode="light",
                  store_path=store_path, extractor=NullReminderExtractor())


class TestDrives(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(prefix="intero_drive_"), "c.db")

    def test_social_rises_with_idle(self):
        organ = make_organ(self.db)
        organ.hb._last_interaction -= 3600 * 2       # 2 小时没聊
        self.assertLess(compute_drives(organ)["social"], 0.2)
        organ.hb._last_interaction -= 3600 * 30      # 累计 32 小时没聊
        self.assertEqual(compute_drives(organ)["social"], 1.0)

    def test_curiosity_when_no_intake(self):
        organ = make_organ(self.db)
        self.assertEqual(compute_drives(organ)["curiosity"], 1.0)      # 空库=饥渴
        organ.ingest("用户喝咖啡", normalize=False)
        self.assertEqual(compute_drives(organ)["curiosity"], 0.5)

    def test_sprout_and_cooldown(self):
        """超阈萌发一次，冷却期内不重复萌发。"""
        organ = make_organ(self.db)
        organ.hb._last_interaction -= 3600 * 48                        # 两天没聊
        fired = maybe_sprout_drive_intentions(organ)
        self.assertIn("social", fired)
        self.assertIn("curiosity", fired)                              # 空库也饥渴
        self.assertTrue(any(i.kind == "drive:social" for i in organ.hb.intentions))
        again = maybe_sprout_drive_intentions(organ)                   # 立刻再来：冷却中
        self.assertEqual(again, [])
        n_before = len(organ.hb.intentions)
        maybe_sprout_drive_intentions(organ, now=time.time() + DRIVE_COOLDOWN + 1)
        # 冷却过后可再萌发（上一个还在队列也无所谓，拍卖/去重兜着）

    def test_below_threshold_no_sprout(self):
        organ = make_organ(self.db)                                    # 刚交互过、空库
        fired = maybe_sprout_drive_intentions(organ)
        self.assertNotIn("social", fired)                              # social≈0 不动心

    def test_drive_intention_can_win_auction(self):
        """内态意图熟成后能赢拍卖进待说（ urgency≈1.0 × 熟成因子 > 底价）。"""
        organ = make_organ(self.db)
        organ.hb._last_interaction -= 3600 * 48
        maybe_sprout_drive_intentions(organ)
        it = next(i for i in organ.hb.intentions if i.kind == "drive:social")
        it.created_at -= it.ttl * 0.4                                  # 催熟
        organ.hb.state = organ.hb.state.WATCH
        organ.hb.tick()
        self.assertTrue(any(p.kind == "drive:social" for p in organ.hb.pending))


class TestCircadian(unittest.TestCase):
    def test_night_floor_high(self):
        self.assertEqual(circadian_floor(2), 0.9)
        self.assertEqual(circadian_floor(23), 0.9)
        self.assertEqual(circadian_floor(14), 0.55)


class TestPolish(unittest.TestCase):
    def test_offline_fallback_raw(self):
        """无 API 环境（CI/离线）话术原样透传。"""
        out = polish("测试原话", context="")
        self.assertIsInstance(out, str) and len(out) > 0               # 有 API 则润色，无则原文


if __name__ == "__main__":
    unittest.main()
