"""主动性闭环的黄金测试：冲动排队 → 持久化 → 见面先说。

设计语义（请求驱动宿主下的诚实形态）：
- 每次工具调用 = interact() + tick()（懒惰心跳，状态走墙钟，不丢 TTL/降档语义）
- 拍卖赢出的意图进 pending（待说事项），下次 recall 置顶送达并清除
- 心跳快照存 sqlite meta 表，新进程（spawn-per-call）恢复后继续跳
"""

import os
import tempfile
import unittest

import numpy as np

from intero.core import Intero
from intero.encoder import TfidfEncoder
from intero.heartbeat import Heartbeat
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer


def make_organ(store_path: str, **hb_kw) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "攀岩", "测试语料"])
    return Intero(
        encoder=enc,
        normalizer=NullNormalizer(),
        memory=TitansMemory(dim=128, hidden=256, depth=3, seed=0),
        store_path=store_path,
        heartbeat=Heartbeat(**hb_kw),
    )


class TestProactiveLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="intero_proactive_")
        self.db = os.path.join(self.tmp, "content.db")

    def test_pending_delivered_once_on_recall(self):
        """高紧迫意图 → 交互一跳即赢拍卖 → recall 置顶送达 → 二次 recall 不再重复。"""
        organ = make_organ(self.db)
        organ.add_intention("remind", "提醒用户周三带攀岩鞋", urgency=0.9, ttl=3600)
        # 出价曲线随时效年龄渐强（测试里墙钟不走，手动把意图催到 lif=0.4 的壮年期）
        organ.hb.intentions[0].created_at -= 3600 * 0.4   # bid = 0.9×0.9 = 0.81 > 底价0.55
        organ.ingest("随便说点什么", normalize=False)      # 交互即心跳：这一拍应赢
        block = organ.recall("攀岩")
        self.assertIn("【待说事项】", block)
        self.assertIn("攀岩鞋", block)
        # 已送达即清除：第二次 recall 不再出现
        self.assertNotIn("【待说事项】", organ.recall("攀岩"))

    def test_silence_beats_weak_intention(self):
        """低紧迫意图压不过沉默底价：不行动、不进待说。"""
        organ = make_organ(self.db)
        organ.add_intention("followup", "无关紧要的追问", urgency=0.3, ttl=3600)
        for _ in range(3):
            organ.tick()
        self.assertEqual(len(organ.hb.pending), 0)
        self.assertNotIn("【待说事项】", organ.recall("咖啡"))

    def test_expired_intention_never_acts(self):
        """TTL 过期的意图被 GC，永远不会进待说。"""
        organ = make_organ(self.db)
        organ.add_intention("remind", "过时的事", urgency=0.9, ttl=1.0)
        organ.hb.intentions[0].created_at -= 100.0      # 手动催老
        organ.tick()
        self.assertEqual(len(organ.hb.intentions), 0)
        self.assertEqual(len(organ.hb.pending), 0)

    def test_state_survives_respawn(self):
        """快照落库：新 Intero 实例（模拟 kimi spawn 新进程）恢复意图与待说。"""
        a = make_organ(self.db)
        a.add_intention("remind", "跨进程也要记住的事", urgency=0.9, ttl=3600)
        a.st.close()
        b = make_organ(self.db)
        self.assertEqual(len(b.hb.intentions), 1)
        self.assertEqual(b.hb.intentions[0].payload, "跨进程也要记住的事")
        b.st.close()

    def test_pending_survives_respawn(self):
        """待说事项跨进程保留（已赢拍卖但还没见面说的，不丢）。"""
        a = make_organ(self.db)
        a.hb.pending.append(
            __import__("intero.heartbeat", fromlist=["Intention"]).Intention(
                kind="remind", payload="已赢未说", urgency=0.9))
        a.st.set_meta_text("heartbeat", __import__("json").dumps(a.hb.snapshot()))
        a.st.close()
        b = make_organ(self.db)
        self.assertEqual([p.payload for p in b.hb.pending], ["已赢未说"])
        b.st.close()


if __name__ == "__main__":
    unittest.main()
