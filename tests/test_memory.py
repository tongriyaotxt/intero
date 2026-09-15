"""tests/test_memory.py — 黄金测试：锁死冠军超参，谁改谁举证。

阈值依据 bench/calibrate.py 的标定结果设定（见 bench/CALIBRATION.md）。
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, ".")
from intero.gates import Gates, GateParams
from intero.memory import TitansMemory
from intero.store import ContentStore

#: 标定台冠军（bench/CALIBRATION.md，2026-09-15 冻结）
REFERENCE = GateParams()  # 默认值即冠军值

DIM, HIDDEN = 128, 1024


def rand_unit(rng, n, dim):
    x = rng.normal(size=(n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def routing(mem, K, V):
    """路由指标：M(k_i) 的余弦得分把 v_i 排进前几名——λ 混合实际消费的就是这个"""
    outs = np.stack([mem.read(k) for k in K])
    outs /= np.linalg.norm(outs, axis=1, keepdims=True)
    S = outs @ V.T
    top1 = float(np.mean(S.argmax(1) == np.arange(len(K))))
    top3 = float(np.mean([i in np.argsort(-S[i])[:3] for i in range(len(K))]))
    return float(np.mean(np.diag(S))), top1, top3


class TestMemoryCore(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_memorize_100_pairs(self):
        """相位1：百对 k-v 背记。阈值按标定结果留安全边距（实测 recall=0.53/top1=0.56/top3=0.90）"""
        mem = TitansMemory(dim=DIM, hidden=HIDDEN, depth=3, gate_params=REFERENCE, seed=0)
        K, V = rand_unit(self.rng, 100, DIM), rand_unit(self.rng, 100, DIM)
        for k, v in zip(K, V):
            mem.write(k, v, force=True)
        recall, top1, top3 = routing(mem, K, V)
        self.assertGreater(recall, 0.35, f"recall={recall:.3f}")
        self.assertGreater(top3, 0.75, f"top3={top3:.3f}")

    def test_gate_saves_writes(self):
        """相位2：门控灌 300 噪声，写入率 < 45%（实测 0.29），抗噪后路由 top3 > 0.40（实测 0.63）"""
        mem = TitansMemory(dim=DIM, hidden=HIDDEN, depth=3, gate_params=REFERENCE, seed=0)
        K, V = rand_unit(self.rng, 100, DIM), rand_unit(self.rng, 100, DIM)
        for k, v in zip(K, V):
            mem.write(k, v, force=True)
        K2, V2 = rand_unit(self.rng, 300, DIM), rand_unit(self.rng, 300, DIM)
        for k, v in zip(K2, V2):
            mem.write(k, v)
        ratio = (mem.writes - 100) / 300
        _, _, top3 = routing(mem, K, V)
        self.assertLess(ratio, 0.45, f"write ratio={ratio:.2f}")
        self.assertGreater(top3, 0.40, f"top3 after noise={top3:.3f}")

    def test_surprise_is_grad_norm(self):
        """Eq.8：瞬时惊讶可计算、非负、重复输入后应下降（学了就不惊讶）"""
        mem = TitansMemory(dim=DIM, hidden=HIDDEN, depth=3, gate_params=REFERENCE, seed=0)
        k, v = rand_unit(self.rng, 1, DIM)[0], rand_unit(self.rng, 1, DIM)[0]
        s1 = mem.surprise(k, v)
        for _ in range(20):
            mem.write(k, v, force=True)
        s2 = mem.surprise(k, v)
        self.assertGreaterEqual(s1, 0)
        self.assertLess(s2, s1, f"s1={s1:.4f} s2={s2:.4f}")

    def test_snapshot_rollback(self):
        """快照机制：save/load 后读出一致"""
        import tempfile, os
        mem = TitansMemory(dim=DIM, hidden=HIDDEN, depth=3, gate_params=REFERENCE, seed=0)
        K, V = rand_unit(self.rng, 10, DIM), rand_unit(self.rng, 10, DIM)
        for k, v in zip(K, V):
            mem.write(k, v, force=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            mem.save(p)
            mem2 = TitansMemory(dim=DIM, hidden=HIDDEN, depth=3, gate_params=REFERENCE, seed=99)
            mem2.load(p)
            for k in K:
                np.testing.assert_allclose(mem.read(k), mem2.read(k), atol=1e-5)


class TestGates(unittest.TestCase):
    def test_theta_monotonic_in_surprise(self):
        g = Gates(REFERENCE)
        for s in [0.01] * 20 + [0.1, 0.2, 0.5, 1.0]:
            g.observe(s)
        self.assertLess(g.theta(0.05), g.theta(1.0))

    def test_alpha_grows_with_redundancy(self):
        g = Gates(REFERENCE)
        self.assertLess(g.alpha(0.0), g.alpha(0.9))

    def test_cold_start_writes(self):
        g = Gates(REFERENCE)
        self.assertTrue(g.should_write(0.001))  # 统计未稳时先写


class TestStore(unittest.TestCase):
    def test_add_retrieve_delete(self):
        import tempfile, os
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            st = ContentStore(os.path.join(d, "s.db"), dim=DIM)
            vecs = rand_unit(rng, 5, DIM)
            ids = [st.add(f"文本{i}", v) for i, v in enumerate(vecs)]
            hits = st.retrieve(vecs[2], topk=1)
            self.assertEqual(hits[0]["id"], ids[2])
            st.delete(ids[2])
            hits = st.retrieve(vecs[2], topk=5)
            self.assertNotIn(ids[2], [h["id"] for h in hits])
            st.close()

    def test_lambda_climbs_as_error_drops(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            st = ContentStore(os.path.join(d, "s.db"), dim=DIM)
            chance = 1.0 / DIM
            self.assertEqual(st.lam, 0.0)
            st.update_lambda(chance * 0.9, chance)   # 误差≈随机水平 → λ≈0
            self.assertLess(st.lam, 0.2)
            st.update_lambda(chance * 0.1, chance)   # 误差远低于随机 → λ 爬升
            self.assertGreater(st.lam, 0.5)
            st.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
