"""MemoryManager：冰糖的轻量级三层记忆管理器。

三层结构（业界做法的轻量融合）：

1. **短期（recent_turns）**：最近 N 轮完整对话。参考 Letta 的 message_buffer_min。
2. **中期（summary）**：累积摘要，LLM 递归更新。参考 Letta 的 summarizer.py。
3. **长期（facts）**：用户画像 KV。参考 Mem0 的事实提取，但用 dict 不用向量库。

特性：
- 后台异步压缩/提取，不阻塞主对话流水线
- JSON 文件持久化，重启可恢复
- 独立的 OpenAI 兼容 client，不依赖 Pipecat 内部 API

典型用法::

    manager = MemoryManager(api_key=..., base_url=..., model=..., persist_path=...)
    manager.load()                       # 启动时加载
    block = manager.build_prompt_block() # 拼到 system prompt
    manager.record_turn("user", "...")   # 每轮对话结束记录
    await manager.wait_for_background_tasks()  # 退出前等待后台任务
    manager.save()                       # 显式保存（record_turn 也会自动 save）
"""

import asyncio
import json
import re
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from .persistence import load_memory, save_memory
from .prompts import FACT_EXTRACTION_PROMPT, SUMMARIZE_PROMPT


# 默认参数
DEFAULT_RECENT_TURNS = 20
DEFAULT_FACT_EXTRACT_EVERY = 5
# 批量压缩阈值：累积多少条溢出消息后，才真正触发一次 LLM 摘要调用。
# 设大了省 token 但摘要更新慢；设小了更新及时但 LLM 调用频繁。
# 经验值 20：1 个回合因 VAD 分段约产生 2-4 条消息，
# 20 条约对应 7-10 个完整回合才触发一次压缩，频率和新鲜度都合适。
DEFAULT_COMPRESS_BATCH = 20

# 摘要/事实提取的 LLM 参数
# 中文摘要要求 300-600 字，但 token 数要留足：
# - 中文一个字约 1.5-2.5 token（视 tokenizer 而定）
# - mimo 等模型可能含 reasoning tokens，会吃 completion 配额
# - 400 实测会出现输出被截断（45 字就断了），故放宽到 1024
_SUMMARY_MAX_TOKENS = 1024
_FACT_MAX_TOKENS = 400
_LLM_TEMPERATURE = 0.3

# 摘要保护：新摘要异常短时（低于此绝对字数），认为 LLM 输出异常，拒绝更新。
# 注意：压缩模式下新摘要比旧摘要短是正常的，不能用比例阈值，改用绝对值。
# 只拦截明显出错的场景（如 LLM 返回了一两句话或空内容）。
_SUMMARY_MIN_CHARS = 50
# 摘要的最大字数上限（防止无限膨胀）
_SUMMARY_MAX_CHARS = 600

# 记忆消息在 context.messages 里的标识前缀，用于定位和更新
MEMORY_MSG_PREFIX = "[记忆背景]"

# 提取 JSON 时去除 markdown 代码块包裹
_CODEBLOCK_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class MemoryManager:
    """冰糖的轻量级三层记忆管理器。

    通过 :meth:`build_prompt_block` 拼接到 system prompt，
    通过 :meth:`record_turn` 在每轮对话结束后追加（异步触发压缩/提取），
    通过 :meth:`save` / :meth:`load` 进行持久化。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        persist_path: str | None = None,
        recent_turns: int = DEFAULT_RECENT_TURNS,
        fact_extract_every: int = DEFAULT_FACT_EXTRACT_EVERY,
        compress_batch: int = DEFAULT_COMPRESS_BATCH,
    ):
        """初始化记忆管理器。

        Args:
            api_key: OpenAI 兼容 API 的密钥。
            base_url: OpenAI 兼容 API 的 base url。
            model: 用于摘要/提取的模型名（建议用便宜的小模型）。
            persist_path: JSON 持久化文件路径；为 None 时不持久化。
            recent_turns: 短期窗口保留的最近对话消息条数（一条 user 或 assistant 算一条）。
            fact_extract_every: 每隔多少个完整对话回合（user+assistant 一回合）
                触发一次事实提取。
            compress_batch: 累积多少条溢出消息后才真正触发一次 LLM 摘要调用。
        """
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        self._model = model
        self._persist_path = persist_path
        self._recent_turns = recent_turns
        self._fact_extract_every = fact_extract_every
        self._compress_batch = compress_batch

        # 三层记忆
        self.recent_turns: list[dict[str, str]] = []
        self.summary: str = ""
        self.facts: dict[str, str] = {}

        # 自上次事实提取以来的「完整回合」计数（按 assistant 计一次）
        self._turns_since_fact_extract: int = 0

        # 暂存的待压缩消息：每次裁剪出的旧消息先进这里，
        # 累积到 _compress_batch 条才真正触发一次 LLM 摘要调用（避免每条都调 LLM）
        self._pending_compress: list[dict[str, str]] = []

        # 压缩串行化锁：保证同一时刻只有一个 _compress_summary 在跑，
        # 避免并发任务读到旧 summary 后互相覆盖。
        # 懒创建，确保在 event loop 内创建。
        self._compress_lock: asyncio.Lock | None = None

        # 后台任务集合（防止 GC 回收）
        self._bg_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def load(self) -> bool:
        """从 JSON 文件加载记忆。

        Returns:
            是否成功加载（文件不存在或解析失败时返回 False）。
        """
        if not self._persist_path:
            return False
        data = load_memory(self._persist_path)
        if data is None:
            return False
        self.recent_turns = data.get("recent_turns", [])
        self.summary = data.get("summary", "")
        self.facts = data.get("facts", {})
        # 恢复待压缩消息（旧版文件无此字段时默认空列表，向后兼容）
        self._pending_compress = data.get("pending_compress", [])
        self._turns_since_fact_extract = data.get("turns_since_fact_extract", 0)
        logger.info(
            f"记忆已加载：{len(self.recent_turns)} 条近对话，"
            f"{len(self.facts)} 条事实，摘要 {len(self.summary)} 字，"
            f"{len(self._pending_compress)} 条待压缩"
        )
        return True

    def save(self) -> None:
        """保存记忆到 JSON 文件。

        采用「写临时文件 + rename」的原子写入方式，避免崩溃时损坏数据。
        同时持久化 ``_pending_compress`` 和回合计数器，确保强制关闭后
        待压缩消息不会丢失。
        """
        if not self._persist_path:
            return
        save_memory(
            self._persist_path,
            {
                "recent_turns": self.recent_turns,
                "summary": self.summary,
                "facts": self.facts,
                # 持久化待压缩消息，防止强杀脚本时丢失
                "pending_compress": self._pending_compress,
                # 持久化计数器，重启后继续累计
                "turns_since_fact_extract": self._turns_since_fact_extract,
            },
        )

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------
    def build_prompt_block(self) -> str:
        """构建可拼接到 system prompt 的记忆区块。

        只包含「非最近」的记忆：facts（用户画像）+ summary（累积摘要）。
        **不**包含 recent_turns，因为 Pipecat 的 LLMContext 自己会累积
        最近对话消息，重复注入会导致 LLM 看到双份内容。

        recent_turns 仅用于内部：作为摘要压缩和事实提取的素材。

        Returns:
            记忆区块文本；无任何记忆时返回空字符串。
        """
        parts: list[str] = []

        if self.facts:
            facts_str = "；".join(
                f"{k}：{v}" for k, v in self.facts.items() if v
            )
            if facts_str:
                parts.append(f"## 关于用户\n{facts_str}")

        if self.summary:
            parts.append(f"## 之前的对话摘要\n{self.summary}")

        return "\n\n".join(parts)

    def build_context_message(self) -> dict[str, str] | None:
        """构建可直接放入 LLMContext.messages 的记忆消息。

        Returns:
            形如 ``{"role": "developer", "content": "..."}`` 的消息字典；
            无记忆时返回 None。
        """
        block = self.build_prompt_block()
        if not block:
            return None
        return {
            "role": "developer",
            "content": f"{MEMORY_MSG_PREFIX}\n以下是关于用户和之前对话的记忆背景，请基于此与用户自然地继续对话：\n\n{block}",
        }

    # ------------------------------------------------------------------
    # 主流程交互
    # ------------------------------------------------------------------
    def record_turn(self, role: str, content: str) -> None:
        """记录一轮对话，并在达到阈值时触发后台压缩/提取。

        设计要点：
        - 短期窗口裁剪是**同步**的（不需要 LLM），保证 recent_turns 不爆。
        - 溢出的消息进入 ``_pending_compress`` 暂存；累积到 batch 才触发 LLM。
        - **只在 assistant 回完检查触发**，user 消息不触发（VAD 分段会产生
          连续多条 user，避免每段都触发）。
        - 摘要压缩任务用 Lock 串行化，避免并发覆盖。

        Args:
            role: ``"user"`` 或 ``"assistant"``。
            content: 该轮的内容文本。
        """
        if not content or not content.strip():
            return
        self.recent_turns.append({"role": role, "content": content.strip()})

        # 短期窗口溢出 -> 同步裁剪，暂存到 _pending_compress（不立即调 LLM）
        if len(self.recent_turns) > self._recent_turns:
            overflow_count = len(self.recent_turns) - self._recent_turns
            self._pending_compress.extend(self.recent_turns[:overflow_count])
            self.recent_turns = self.recent_turns[overflow_count:]

        # 每轮都先保存，防止丢数据
        self.save()

        # 只有 assistant 回完（一个完整回合结束）才检查是否触发批量压缩
        # 避免 user 分段说话时每段都触发一次
        if role == "assistant":
            if len(self._pending_compress) >= self._compress_batch:
                batch = self._pending_compress
                self._pending_compress = []
                self._spawn_task(self._compress_summary_safe(batch))

            # 事实提取计数（每 N 个完整回合触发）
            self._turns_since_fact_extract += 1
            if self._turns_since_fact_extract >= self._fact_extract_every:
                self._turns_since_fact_extract = 0
                self._spawn_task(self._extract_facts())

    async def flush_pending(self) -> None:
        """强制把 _pending_compress 里的待压缩消息立刻压缩。

        用于优雅退出，确保 session 结束时所有溢出消息都已并入 summary，
        不会丢在内存里。

        若当前没有 pending，则什么都不做。
        """
        if not self._pending_compress:
            return
        batch = self._pending_compress
        self._pending_compress = []
        await self._compress_summary_safe(batch)

    def resume_pending(self) -> None:
        """重启后处理上次遗留的待压缩消息。

        当脚本被强制关闭（kill -9、关终端）时，``_pending_compress`` 里的
        消息虽然已通过 :meth:`save` 持久化到磁盘，但还没被压缩进 summary。
        本方法在重启后调用，把遗留的 pending 消息提交到后台压缩任务，
        确保信息不会因为重启而永远滞留在 pending 队列里。

        若当前没有 pending，则什么都不做。
        """
        if not self._pending_compress:
            return
        batch = self._pending_compress
        self._pending_compress = []
        logger.info(f"[记忆] 重启后恢复 {len(batch)} 条遗留待压缩消息，提交后台压缩")
        self._spawn_task(self._compress_summary_safe(batch))

    async def wait_for_background_tasks(self, timeout: float = 5.0) -> None:
        """等待所有后台任务完成（用于优雅退出）。

        Args:
            timeout: 最大等待时间（秒）。
        """
        if not self._bg_tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._bg_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"记忆后台任务等待超时（{timeout}s），强制结束")
        self._bg_tasks.clear()

    # ------------------------------------------------------------------
    # 后台任务实现
    # ------------------------------------------------------------------
    def _spawn_task(self, coro: Any) -> None:
        """启动后台任务并跟踪。

        Args:
            coro: 待执行的协程。
        """
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _get_compress_lock(self) -> asyncio.Lock:
        """懒创建压缩锁，确保在 event loop 上下文内创建。

        Returns:
            asyncio.Lock 实例。
        """
        if self._compress_lock is None:
            self._compress_lock = asyncio.Lock()
        return self._compress_lock

    async def _compress_summary_safe(self, to_compress: list[dict[str, str]]) -> None:
        """带锁的压缩入口，保证多个压缩任务串行执行。

        避免并发场景：任务 A 读到旧 summary 开始 LLM 调用 -> 任务 B 也读到旧
        summary -> A 写新 summary -> B 完成后覆盖 A 的结果。加锁后必须排队。

        Args:
            to_compress: 待压缩的消息列表。
        """
        async with self._get_compress_lock():
            await self._compress_summary(to_compress)

    async def _compress_summary(self, to_compress: list[dict[str, str]]) -> None:
        """将一段已裁剪的旧对话增量合并进累积摘要。

        注意：调用方应通过 :meth:`_compress_summary_safe` 获取锁后再调用本方法。

        Args:
            to_compress: 已经从 recent_turns 中拿出来的待压缩消息列表。
        """
        if not to_compress:
            return
        try:
            messages_text = "\n".join(
                f"{'用户' if t['role'] == 'user' else '冰糖'}：{t['content']}"
                for t in to_compress
            )

            prompt = SUMMARIZE_PROMPT.format(
                existing_summary=self.summary or "（暂无）",
                new_messages=messages_text,
            )

            logger.info(f"[记忆] 开始压缩 {len(to_compress)} 条对话到摘要（当前摘要 {len(self.summary)} 字）...")
            new_summary = await self._call_llm(prompt, _SUMMARY_MAX_TOKENS)
            if new_summary:
                new_summary = new_summary.strip()
                old_len = len(self.summary)

                # 保护机制：只拦截 LLM 明显出错的场景（输出极短）。
                # 压缩模式下新摘要比旧摘要短是正常的——prompt 允许精简旧内容，
                # 所以不能用比例阈值，改用绝对值。低于最小字数时拒绝更新，
                # 保留旧摘要（新对话信息通过 recent_turns 兜底，不会真的丢）。
                # 首次压缩（无旧摘要）时不检查——第一条摘要短是正常的。
                if self.summary and len(new_summary) < _SUMMARY_MIN_CHARS:
                    logger.warning(
                        f"[记忆] 新摘要异常短（{len(new_summary)} 字 < "
                        f"{_SUMMARY_MIN_CHARS}），LLM 输出可能异常，"
                        f"保留旧摘要不改"
                    )
                else:
                    # 正常截断到上限（防止极端情况下超长）
                    if len(new_summary) > _SUMMARY_MAX_CHARS:
                        new_summary = new_summary[:_SUMMARY_MAX_CHARS]
                    self.summary = new_summary

                logger.info(
                    f"[记忆] 摘要已更新（{old_len} -> {len(self.summary)} 字，"
                    f"压缩了 {len(to_compress)} 条）"
                )

            self.save()
        except Exception as e:
            logger.warning(f"[记忆] 压缩摘要失败: {e}")

    async def _extract_facts(self) -> None:
        """从最近对话中提取用户事实 KV（后台异步执行）。

        参考 Mem0 的事实提取，但用固定字段 dict 代替向量库。
        """
        try:
            # 取最近若干条作为提取素材（一回合 = 2 条消息）
            window_size = self._fact_extract_every * 2
            window = self.recent_turns[-window_size:]
            if not window:
                return

            messages_text = "\n".join(
                f"{'用户' if t['role'] == 'user' else '冰糖'}：{t['content']}"
                for t in window
            )

            existing_facts_str = (
                "\n".join(f"- {k}：{v}" for k, v in self.facts.items())
                or "（暂无）"
            )

            prompt = FACT_EXTRACTION_PROMPT.format(
                existing_facts=existing_facts_str,
                recent_messages=messages_text,
            )

            logger.info("[记忆] 开始提取用户事实...")
            response = await self._call_llm(prompt, _FACT_MAX_TOKENS)
            if not response:
                return

            cleaned = _CODEBLOCK_RE.sub("", response.strip())
            new_facts = json.loads(cleaned)
            if not isinstance(new_facts, dict):
                logger.debug("[记忆] 事实提取结果非对象，跳过")
                return

            updated_keys: list[str] = []
            for k, v in new_facts.items():
                if isinstance(v, str) and v.strip() and self.facts.get(k) != v:
                    self.facts[k] = v.strip()
                    updated_keys.append(k)

            if updated_keys:
                logger.info(f"[记忆] 事实已更新字段：{updated_keys}")
                self.save()
            else:
                logger.debug("[记忆] 无新事实可更新")
        except json.JSONDecodeError as e:
            logger.warning(f"[记忆] 事实 JSON 解析失败: {e}")
        except Exception as e:
            logger.warning(f"[记忆] 提取事实失败: {e}")

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------
    async def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """调用 OpenAI 兼容接口获取文本响应。

        Args:
            prompt: 已格式化的用户 prompt。
            max_tokens: 最大输出 token 数。

        Returns:
            LLM 返回的文本；失败时返回空字符串。
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=_LLM_TEMPERATURE,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"[记忆] LLM 调用失败: {e}")
            return ""
