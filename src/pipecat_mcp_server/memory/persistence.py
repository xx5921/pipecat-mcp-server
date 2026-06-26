"""记忆模块的 JSON 文件持久化。

提供两个简单函数：``save_memory`` 和 ``load_memory``。
单用户场景下 JSON 足够；如未来需要多用户或并发写入，可换 SQLite。
"""

import json
import os
from typing import Any

from loguru import logger


def save_memory(path: str, data: dict[str, Any]) -> None:
    """将记忆数据保存到 JSON 文件。

    Args:
        path: JSON 文件路径；父目录会被自动创建。
        data: 待保存的字典数据。
    """
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_memory(path: str) -> dict[str, Any] | None:
    """从 JSON 文件加载记忆数据。

    Args:
        path: JSON 文件路径。

    Returns:
        记忆字典；文件不存在或解析失败时返回 ``None``。
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"读取记忆文件失败: {e}")
        return None
