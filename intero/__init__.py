"""intero — Titans 式测试时神经记忆 + 自主心跳（竖片）。

设计文档：tech-base/21、22（工作区）。论文：arXiv:2501.00663。
"""

from .encoder import Encoder, TfidfEncoder, STEncoder, APIEncoder, best_available, probe
from .gates import Gates, GateParams
from .memory import TitansMemory
from .store import ContentStore

__all__ = [
    "Encoder", "TfidfEncoder", "STEncoder", "APIEncoder", "best_available", "probe",
    "Gates", "GateParams", "TitansMemory", "ContentStore",
]
__version__ = "0.1.0"
