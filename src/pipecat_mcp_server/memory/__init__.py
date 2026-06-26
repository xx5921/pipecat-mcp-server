"""冰糖的轻量级三层记忆管理子包。

提供 ``MemoryManager`` 类，负责短期对话窗口、中期摘要、长期事实 KV 的维护，
以及 JSON 文件持久化。所有 LLM 调用走后台异步任务，不阻塞主对话流水线。
"""

from .manager import DEFAULT_FACT_EXTRACT_EVERY, DEFAULT_RECENT_TURNS, MemoryManager

__all__ = ["MemoryManager", "DEFAULT_RECENT_TURNS", "DEFAULT_FACT_EXTRACT_EVERY"]
