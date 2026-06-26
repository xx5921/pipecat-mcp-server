"""唤醒词相关的纯函数工具。

包括文本归一化（去标点空白）、唤醒词匹配、尾部 user 消息合并。
"""

import re

_PUNCT_RE = re.compile(r"[]\s，。！？、；：""''（）《》【】,.!?;:\"'(){}[]+")


def normalize(text: str) -> str:
    """移除中英文标点和空白，用于唤醒词匹配。

    Args:
        text: 原始文本。

    Returns:
        去除标点空白后的文本。
    """
    return _PUNCT_RE.sub("", text)


def find_wake_word(text: str, wake_words: list[str]) -> str | None:
    """在文本中查找第一个匹配的唤醒词（忽略标点空白）。

    Args:
        text: 待检测的文本。
        wake_words: 唤醒词列表。

    Returns:
        匹配到的唤醒词；未匹配返回 None。
    """
    normalized = normalize(text)
    for w in wake_words:
        if w in normalized:
            return w
    return None


def merge_tail_user_messages(context) -> None:
    """合并 LLMContext 尾部连续的多条 user 消息为一条。

    VAD 可能把长句切成多段，每段生成一条 ``{"role": "user"}`` 消息。
    此函数将尾部连续的 user 消息合并，避免 LLM 上下文碎片化。

    Args:
        context: LLMContext 实例。
    """
    messages = context.messages
    if len(messages) < 2:
        return

    # 从尾部扫描，找到连续 user 消息的起始位置
    tail_start = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            tail_start = i + 1
            break
    if tail_start is None:
        tail_start = 0

    user_msgs = messages[tail_start:]
    if len(user_msgs) <= 1:
        return

    merged_content = " ".join(m.get("content", "") for m in user_msgs)
    context.set_messages(
        messages[:tail_start] + [{"role": "user", "content": merged_content}]
    )
