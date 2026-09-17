"""M5 夜间时钟的黄金测试：回放 / 去重 / 晋升 / 健康线。"""

import os
import tempfile
import unittest

import numpy as np

from intero.dream import dream
from intero.memory import TitansMemory
from intero.store import ContentStore


def _unit(rng, dim=64):
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestDream(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.dim = 64
        self.tmp = tempfile.mkdtemp(prefix="intero_dream_test_")
        self.mem = TitansMemory(dim=self.dim, hidden=256, depth=3, seed=0)
        self.st = ContentStore(os.path.join(self.tmp, "c.db"), dim=self.dim)

    def tearDown(self):
        self.st.close()

    def test_replay_promote_and_health(self):
        # 写入 12 条事实（force 入库存，模拟白天门控后的内容库）
        facts = [f"事实{i}" for i in range(12)]
        for t in facts:
            v = _unit(self.rng, self.dim)
            self.mem.write(v, v, force=True)
            self.st.add(t, v, kind="fact")
        report = dream(self.mem, self.st)
        self.assertEqual(report["回放"], 12)                 # 全部回放
        self.assertGreater(report["晋升"], 0)                # 有晋升
        kinds = [it["kind"] for it in self.st.items()]
        self.assertIn("consolidated", kinds)
        self.assertTrue(report.get("健康", True))            # 回放后误差未恶化

    def test_dedup_archives_later_copy(self):
        v = _unit(self.rng, self.dim)
        self.st.add("原始事实", v, kind="fact")
        self.st.add("原始事实", v.copy(), kind="fact")       # 完全重复
        other = _unit(self.rng, self.dim)
        self.st.add("不相干事实", other, kind="fact")
        report = dream(self.mem, self.st)
        self.assertEqual(report["去重"], 1)
        kinds = sorted(it["kind"] for it in self.st.items())
        self.assertIn("dedup", kinds)

    def test_empty_store_is_noop(self):
        report = dream(self.mem, self.st)
        self.assertEqual(report["回放"], 0)


if __name__ == "__main__":
    unittest.main()
