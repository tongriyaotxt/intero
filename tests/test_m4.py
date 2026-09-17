"""M4 的黄金测试：注入装配 + Intero 门面 + MCP 工具函数。"""

import unittest

import numpy as np

from intero.inject import assemble


class TestInject(unittest.TestCase):
    def test_priority_and_conflict_mark(self):
        results = [
            {"text": "普通事实B", "kind": "fact", "score": 0.9},
            {"text": "晋升事实A", "kind": "consolidated", "score": 0.5},
            {"text": "矛盾事实C", "kind": "conflict", "score": 0.99},
            {"text": "噪声D", "kind": "noise", "score": 0.99},
        ]
        block = assemble(results, topk=5)
        lines = block.split("\n")
        self.assertIn("晋升事实A", lines[1])          # consolidated 排最前
        self.assertIn("⚠️", block)                     # conflict 带标注
        self.assertNotIn("噪声D", block)               # noise 不上墙

    def test_budget_truncation(self):
        results = [{"text": "很长的条目" * 50, "kind": "fact", "score": 0.9}]
        self.assertEqual(assemble(results, budget_chars=50), "")  # 宁缺毋滥

    def test_empty(self):
        self.assertEqual(assemble([]), "")


class TestCoreFacade(unittest.TestCase):
    """门面级集成测试：小记忆 + TF-IDF（不依赖外部模型/网络）。"""

    def test_ingest_recall_status(self):
        from intero.core import Intero
        from intero.encoder import TfidfEncoder
        from intero.memory import TitansMemory
        from intero.normalize import NullNormalizer

        enc = TfidfEncoder(dim=128)
        organ = Intero(
            encoder=enc,
            normalizer=NullNormalizer(),
            memory=TitansMemory(dim=128, hidden=256, depth=3, seed=0),
        )
        corpus = ["小李喝咖啡不加糖", "老王住在海淀区", "数据库索引怎么优化"]
        enc.partial_fit(corpus + ["小李喝咖啡加什么"])
        for t in corpus:
            organ.ingest(t, normalize=False)
        res = organ.retrieve("小李喝咖啡加什么", topk=2)
        self.assertTrue(any("小李" in r["text"] for r in res))
        block = organ.recall("小李喝咖啡加什么", topk=2)
        self.assertIsInstance(block, str)
        st = organ.status()
        self.assertIn("写入", st)
        self.assertGreaterEqual(st["写入"], 1)


class TestMcpTools(unittest.TestCase):
    def test_tool_functions_importable(self):
        from intero import mcp_server

        for fn in ("memory_write", "memory_recall", "memory_status"):
            self.assertTrue(callable(getattr(mcp_server, fn)))


if __name__ == "__main__":
    unittest.main()
