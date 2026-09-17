"""服务化的黄金测试：HTTP 服务 + 瘦客户端 + 锁并发 + 服务不在时的快速失败。"""

import os
import tempfile
import threading
import time
import unittest

from intero.client import RemoteOrgan
from intero.core import Intero
from intero.encoder import TfidfEncoder
from intero.intents import NullReminderExtractor
from intero.memory import TitansMemory
from intero.normalize import NullNormalizer
from intero.service import serve_in_thread


def make_organ(store_path: str) -> Intero:
    enc = TfidfEncoder(dim=128)
    enc.partial_fit(["咖啡", "攀岩", "测试语料"])
    return Intero(
        encoder=enc,
        normalizer=NullNormalizer(),
        memory=TitansMemory(dim=128, hidden=256, depth=3, seed=0),
        store_path=store_path,
        extractor=NullReminderExtractor(),
    )


class TestService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="intero_svc_")
        cls.organ = make_organ(os.path.join(cls.tmp, "content.db"))
        cls.srv = serve_in_thread(cls.organ, port=0)           # 0 = 随机空闲端口
        cls.port = cls.srv.server_address[1]
        cls.remote = RemoteOrgan(base=f"http://127.0.0.1:{cls.port}")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.organ.st.close()

    def test_alive(self):
        self.assertTrue(self.remote.is_alive())

    def test_dead_service_fails_fast(self):
        dead = RemoteOrgan(base="http://127.0.0.1:1")          # 端口 1 必死
        t0 = time.time()
        self.assertFalse(dead.is_alive())
        self.assertLess(time.time() - t0, 2.0)                 # 快速失败，不挂起

    def test_write_recall_roundtrip(self):
        r = self.remote.ingest("小李喝咖啡不加糖")
        self.assertEqual(r["written"], 1)
        self.assertIn("小李", self.remote.recall("小李喝咖啡加什么"))

    def test_status_and_intention_and_tick(self):
        st = self.remote.status()
        self.assertIn("库存", st)
        self.remote.add_intention("remind", "测试意图", urgency=0.9, ttl=600)
        r = self.remote.tick()
        self.assertIn("意图队列", r)

    def test_dream_endpoint(self):
        r = self.remote.dream_cycle()
        self.assertIn("意图自生", r)

    def test_concurrent_writes_no_loss(self):
        """锁保护：多线程并发写不丢（服务化版 spawn 竞态回归）。"""
        errors = []

        def w(n):
            try:
                self.remote.ingest(f"并发事实{n}")
            except Exception as e:                             # noqa: BLE001
                errors.append(e)

        ts = [threading.Thread(target=w, args=(i,)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errors, [])
        texts = [it["text"] for it in self.organ.st.items()]
        for i in range(8):
            self.assertIn(f"并发事实{i}", texts)


if __name__ == "__main__":
    unittest.main()
