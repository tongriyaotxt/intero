"""daemon 的黄金测试：主动送达路由 + daemon 跳拍不污染交互时钟。"""

import io
import os
import tempfile
import unittest

from intero.core import Intero
from intero.daemon import deliver_due, run
from intero.encoder import TfidfEncoder
from intero.heartbeat import HeartState, Heartbeat, Intention
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
    )


class TestDaemonDelivery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="intero_daemon_")
        self.db = os.path.join(self.tmp, "content.db")
        self.sent = []

    def notify(self, intention, voice=False):
        self.sent.append(intention.payload)

    def test_engaged_holds_delivery(self):
        """用户正在聊（ENGAGED）：daemon 不主动送，留给对话内 recall。"""
        organ = make_organ(self.db)
        organ._beat()                                    # 模拟用户刚交互 → ENGAGED
        organ.hb.pending.append(Intention(kind="remind", payload="先攒着"))
        self.assertEqual(deliver_due(organ, notify_fn=self.notify), 0)
        self.assertEqual(self.sent, [])
        self.assertEqual(len(organ.hb.pending), 1)       # 还在待说里

    def test_watch_delivers_proactively(self):
        """用户离开（WATCH）：立即主动送达并清除。"""
        organ = make_organ(self.db)
        organ.hb.state = HeartState.WATCH
        organ.hb.pending.append(Intention(kind="remind", payload="主动找你说话"))
        self.assertEqual(deliver_due(organ, notify_fn=self.notify), 1)
        self.assertEqual(self.sent, ["主动找你说话"])
        self.assertEqual(len(organ.hb.pending), 0)
        # 送达后快照落库：新进程看不到已送的
        organ.st.close()
        organ2 = make_organ(self.db)
        self.assertEqual(len(organ2.hb.pending), 0)
        organ2.st.close()

    def test_daemon_tick_does_not_interact(self):
        """daemon 跳拍不得刷新交互时钟（否则状态永远 ENGAGED，永远不主动）。"""
        organ = make_organ(self.db)
        organ.hb._last_interaction -= 120.0              # 2 分钟无交互 → 应降 WATCH
        r = organ.daemon_tick()
        self.assertEqual(r["状态"], "watch")
        # 若误调 interact()，状态会被拉回 engaged

    def test_run_once_picks_up_mcp_side_intention(self):
        """跨进程吸收：MCP 侧写入的意图，daemon 每拍 reload 后能看到并熟成行动。"""
        a = make_organ(self.db)
        a.add_intention("remind", "来自 MCP 的意图", urgency=1.0, ttl=300)
        a.st.close()
        organ = make_organ(self.db)
        # 催老意图到熟成区 + 用户已离开（WATCH），daemon 跑一圈应主动送达
        import json
        organ.reload_heartbeat()
        organ.hb.intentions[0].created_at -= 300 * 0.4
        organ.hb._last_interaction -= 120.0
        organ.save_heartbeat()
        run(organ, once=True, notify_fn=self.notify, out=io.StringIO())
        self.assertEqual(self.sent, ["来自 MCP 的意图"])


if __name__ == "__main__":
    unittest.main()
