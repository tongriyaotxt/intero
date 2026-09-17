"""意图自生的黄金测试：萌发 / 去重 / 死线 TTL / 过期不萌发 / 检查标记不重复付费 / DREAM 周期联动。"""

import io
import os
import tempfile
import time
import unittest

from intero.core import Intero
from intero.daemon import run
from intero.encoder import TfidfEncoder
from intero.heartbeat import HeartState
from intero.intents import NullReminderExtractor, parse_reminders
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer


class StubExtractor:
    """离线桩：对含"交稿"的事实返回一条带死线的提醒。"""

    name = "stub"

    def __init__(self, reminders):
        self._rems = reminders
        self.calls = 0

    def extract(self, facts):
        self.calls += 1
        return list(self._rems)


def make_organ(store_path: str, extractor) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["交稿", "知乎", "攀岩", "测试语料"])
    return Intero(
        encoder=enc,
        normalizer=NullNormalizer(),
        memory=TitansMemory(dim=128, hidden=256, depth=3, seed=0),
        store_path=store_path,
        extractor=extractor,
    )


class TestIntentionSprout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="intero_sprout_")
        self.db = os.path.join(self.tmp, "content.db")

    def test_parse_reminders_robust(self):
        """解析器：正常数组 / 杂质包裹 / 坏 JSON / 坏死线都能兜住。"""
        good = parse_reminders(
            '前面废话[{"payload": "交稿", "deadline": "2026-09-22T15:00", "urgency": 0.9}]后面')
        self.assertEqual(good[0]["payload"], "交稿")
        self.assertIsNotNone(good[0]["deadline_ts"])
        self.assertEqual(parse_reminders("没有数组"), [])
        self.assertEqual(parse_reminders("[{坏 json]"), [])
        bad_dl = parse_reminders('[{"payload": "x", "deadline": "下周二", "urgency": "高"}]')
        self.assertIsNone(bad_dl[0]["deadline_ts"])
        self.assertEqual(bad_dl[0]["urgency"], 0.8)          # 坏 urgency 回退默认

    def test_sprout_with_deadline(self):
        """带死线的提醒 → 注册 sprout 意图，TTL ≈ 距死线的秒数。"""
        dl = time.time() + 7200
        ex = StubExtractor([{"payload": "知乎初稿明天下午3点截止", "deadline_ts": dl, "urgency": 0.9}])
        organ = make_organ(self.db, ex)
        organ.ingest("下周二下午3点前要交知乎文章初稿", normalize=False)
        r = organ.derive_intentions()
        self.assertEqual(r["萌发"], 1)
        it = organ.hb.intentions[0]
        self.assertTrue(it.kind.startswith("sprout"))
        self.assertAlmostEqual(it.ttl, 7200, delta=60)
        self.assertEqual(it.urgency, 0.9)

    def test_no_double_sprout_and_no_repay(self):
        """同一事实不重复萌发（payload 去重 + 检查标记不重复付 LLM 费）。"""
        ex = StubExtractor([{"payload": "同一件事", "deadline_ts": None, "urgency": 0.8}])
        organ = make_organ(self.db, ex)
        organ.ingest("每周三晚上去攀岩馆", normalize=False)
        organ.derive_intentions()
        organ.derive_intentions()                            # 第二次：事实已标记检查
        self.assertEqual(ex.calls, 1)                        # LLM 只被调一次
        self.assertEqual(len(organ.hb.intentions), 1)        # 意图不重复

    def test_expired_deadline_not_sprouted(self):
        """死线已过的事项不萌发。"""
        ex = StubExtractor([{"payload": "昨天的会", "deadline_ts": time.time() - 100, "urgency": 1.0}])
        organ = make_organ(self.db, ex)
        organ.ingest("昨天有个会", normalize=False)
        self.assertEqual(organ.derive_intentions()["萌发"], 0)
        self.assertEqual(len(organ.hb.intentions), 0)

    def test_null_extractor_safe_noop(self):
        """离线（无 LLM）：不萌发，系统照跑。"""
        organ = make_organ(self.db, NullReminderExtractor())
        organ.ingest("明天要交稿", normalize=False)
        self.assertEqual(organ.derive_intentions()["萌发"], 0)

    def test_dream_transition_sprouts(self):
        """daemon 进入 DREAM 的第一拍触发夜间周期，意图在睡梦中萌发。"""
        dl = time.time() + 7200
        ex = StubExtractor([{"payload": "睡梦中想到的提醒", "deadline_ts": dl, "urgency": 0.9}])
        organ = make_organ(self.db, ex)
        organ.ingest("明天下午3点交知乎初稿", normalize=False)
        organ.hb.sleep()                                     # 显式入睡
        organ.save_heartbeat()
        sent = []
        run(organ, once=True, notify_fn=lambda i, voice=False: sent.append(i.payload),
            out=io.StringIO())
        self.assertTrue(any(it.kind.startswith("sprout") for it in organ.hb.intentions))


if __name__ == "__main__":
    unittest.main()
