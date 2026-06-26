"""独立语音助手的主流程（run_bot）。

负责：
- 创建 STT/TTS/LLM 服务
- 创建并初始化 MemoryManager，启动时从磁盘加载
- 组装 Pipecat 流水线
- 注册客户端连接/断开、用户/助手轮次结束的事件回调
- 在每个轮次回调里同步记忆 + 维护 context 中的记忆背景消息
- 退出时等待后台任务 + 保存记忆
"""

import asyncio
import os
import random

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport
from pipecat.workers.runner import WorkerRunner
from zhconv import convert as zh_convert

from pipecat_mcp_server.memory import MemoryManager
from pipecat_mcp_server.memory.manager import MEMORY_MSG_PREFIX

from . import constants as C
from .default_tools import BOT_TOOLS
from .mcp_loader import cleanup_mcp, load_mcp_tools
from .services import (
    create_llm_service,
    create_stt_service,
    create_tts_service,
    get_llm_credentials,
)
from .wake_word import find_wake_word, normalize

load_dotenv(override=True)


def _build_memory_manager() -> MemoryManager | None:
    """根据环境变量构建 MemoryManager。

    当 ``PIPECAT_MEMORY_ENABLED`` 显式为 ``"0"`` / ``"false"`` 时返回 None，
    否则构建并加载已有记忆。

    Returns:
        配置完成的 MemoryManager；用户禁用时返回 None。
    """
    enabled = os.environ.get("PIPECAT_MEMORY_ENABLED", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        logger.info("[记忆] 已通过 PIPECAT_MEMORY_ENABLED=0 禁用记忆模块")
        return None

    api_key, base_url, main_model = get_llm_credentials()
    # 记忆模块默认复用主对话模型；可用 PIPECAT_LLM_MEMORY_MODEL 指定更便宜的模型
    memory_model = (
        os.environ.get("PIPECAT_LLM_MEMORY_MODEL", C.DEFAULT_MEMORY_MODEL).strip()
        or main_model
    )

    memory_dir = os.environ.get("PIPECAT_MEMORY_DIR", C.DEFAULT_MEMORY_DIR)
    memory_file = os.environ.get("PIPECAT_MEMORY_FILE", C.DEFAULT_MEMORY_FILE)
    persist_path = os.path.join(memory_dir, memory_file)

    recent_turns = int(os.environ.get("PIPECAT_MEMORY_RECENT_TURNS", C.DEFAULT_MEMORY_RECENT_TURNS))
    fact_every = int(os.environ.get("PIPECAT_MEMORY_FACT_EVERY", C.DEFAULT_MEMORY_FACT_EVERY))
    compress_batch = int(os.environ.get("PIPECAT_MEMORY_COMPRESS_BATCH", C.DEFAULT_COMPRESS_BATCH))

    manager = MemoryManager(
        api_key=api_key,
        base_url=base_url,
        model=memory_model,
        persist_path=persist_path,
        recent_turns=recent_turns,
        fact_extract_every=fact_every,
        compress_batch=compress_batch,
    )
    loaded = manager.load()
    if loaded:
        logger.info(f"[记忆] 已从 {persist_path} 恢复记忆")
    else:
        logger.info(f"[记忆] 无历史记忆文件，将从头开始（将保存到 {persist_path}）")
    return manager


def _sync_memory_to_context(memory_manager: MemoryManager, context: LLMContext) -> None:
    """将记忆同步为 context.messages 中的一条 developer 消息。

    通过 ``MEMORY_MSG_PREFIX`` 前缀定位已有记忆消息：
    - 找到 -> 原地更新
    - 未找到 -> 插入到最前面

    Args:
        memory_manager: 记忆管理器实例。
        context: Pipecat LLMContext。
    """
    msg = memory_manager.build_context_message()
    messages = context.messages
    if msg is None:
        # 没有记忆可注入：若历史上有记忆消息，也保留原状（不影响下游）
        return
    if messages and isinstance(messages[0].get("content"), str) and messages[0]["content"].startswith(MEMORY_MSG_PREFIX):
        messages[0] = msg
    else:
        messages.insert(0, msg)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """组装并运行机器人的主逻辑。

    Args:
        transport: Pipecat 传输实例。
        runner_args: Runner 配置参数。
    """
    logger.info("正在启动独立语音助手（冰糖）...")

    # -----------------------------------------------------------------
    # 唤醒词状态
    # -----------------------------------------------------------------
    raw = os.environ.get("PIPECAT_WAKE_WORD", "").strip()
    wake_words = [w.strip() for w in raw.split(",") if w.strip()]
    awake = not bool(wake_words)  # 无唤醒词时始终唤醒
    wake_timeout_secs = float(os.environ.get("PIPECAT_WAKE_TIMEOUT", "60"))
    awake_timeout_task: asyncio.Task | None = None

    async def schedule_awake_timeout():
        """在超时后让助手回到睡眠状态。"""
        nonlocal awake_timeout_task, awake

        async def _timeout():
            nonlocal awake
            try:
                await asyncio.sleep(wake_timeout_secs)
                logger.info(f"唤醒超时（{wake_timeout_secs}s），回到睡眠状态")
                awake = False
            except asyncio.CancelledError:
                pass

        if awake_timeout_task:
            awake_timeout_task.cancel()
        awake_timeout_task = asyncio.create_task(_timeout())

    def cancel_awake_timeout():
        """取消当前的唤醒超时任务。"""
        nonlocal awake_timeout_task
        if awake_timeout_task:
            awake_timeout_task.cancel()
            awake_timeout_task = None

    # -----------------------------------------------------------------
    # 记忆模块
    # -----------------------------------------------------------------
    memory_manager = _build_memory_manager()

    # -----------------------------------------------------------------
    # MCP 工具 + 内置工具
    # -----------------------------------------------------------------
    mcp_server_urls = [
        u.strip()
        for u in os.environ.get("PIPECAT_MCP_SERVERS", C.DEFAULT_MCP_SERVERS).split(",")
        if u.strip()
    ]
    mcp_tools = await load_mcp_tools(mcp_server_urls)
    all_tools = BOT_TOOLS + mcp_tools
    logger.info(f"总共注册 {len(all_tools)} 个工具（内置 {len(BOT_TOOLS)} + MCP {len(mcp_tools)}）")

    # -----------------------------------------------------------------
    # 服务 & 流水线
    # -----------------------------------------------------------------
    stt = create_stt_service()
    tts = create_tts_service()
    llm = create_llm_service()

    context = LLMContext(tools=all_tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.2,
                min_volume=0.6,
            )),
        ),
    )

    # 流水线：STT → 用户聚合器 → LLM → TTS → 传输输出 → 助手聚合器
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # -----------------------------------------------------------------
    # 事件回调
    # -----------------------------------------------------------------
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("客户端已连接")
        # 先把记忆背景同步到 context（若有）
        if memory_manager is not None:
            # 重启后如果有遗留的待压缩消息（上次强杀脚本导致），立即提交后台压缩
            memory_manager.resume_pending()
            _sync_memory_to_context(memory_manager, context)
        context.add_message(
            {"role": "developer", "content": "用一两句活泼的口语向用户打个招呼，自报家门说你叫冰糖，问问对方今天怎么样。"}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("客户端已断开")
        cancel_awake_timeout()
        await worker.cancel()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        nonlocal awake

        if not message.content:
            return

        text = zh_convert(message.content, "zh-cn")
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"转录: {timestamp}用户: {text}")

        if not awake and wake_words:
            # 睡眠状态：检查唤醒词
            matched = find_wake_word(text, wake_words)
            if matched:
                logger.info(f"唤醒词 '{matched}' 检测到: '{text}'")
                awake = True
                await schedule_awake_timeout()
                remaining = normalize(text).replace(matched, "", 1).strip()
                if remaining:
                    # 唤醒词 + 问题：更新最后一条 user 消息为剩余内容
                    if context.messages:
                        last = context.messages[-1]
                        if last.get("role") == "user":
                            last["content"] = remaining
                    # 被唤醒并且带了问题：记入记忆
                    if memory_manager is not None:
                        memory_manager.record_turn("user", remaining)
                        _sync_memory_to_context(memory_manager, context)
                else:
                    # 仅唤醒词：发送随机问候
                    greeting = random.choice(C.GREETINGS)
                    logger.info(f"播放唤醒问候: '{greeting}'")
                    context.add_message(
                        {"role": "developer", "content": f"请用语音说出: {greeting}"}
                    )
                    await worker.queue_frames([LLMRunFrame()])
            else:
                logger.debug(f"睡眠中忽略语音: '{text}'")
                # 移除最后一条 user 消息，避免 LLM 响应
                if context.messages and context.messages[-1].get("role") == "user":
                    context.messages.pop()
        else:
            # 唤醒状态：正常处理
            if wake_words:
                cancel_awake_timeout()
                await schedule_awake_timeout()
            # 记入记忆
            if memory_manager is not None:
                memory_manager.record_turn("user", text)
                _sync_memory_to_context(memory_manager, context)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}助手: {message.content}"
        logger.info(f"转录: {line}")
        # 助手回完 -> 记入记忆，可能触发后台事实提取
        if memory_manager is not None and message.content:
            memory_manager.record_turn("assistant", message.content)
            _sync_memory_to_context(memory_manager, context)

    # -----------------------------------------------------------------
    # 运行
    # -----------------------------------------------------------------
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()

    # -----------------------------------------------------------------
    # 清理
    # -----------------------------------------------------------------
    cancel_awake_timeout()
    if memory_manager is not None:
        logger.info("[记忆] 等待后台任务完成并保存...")
        await memory_manager.wait_for_background_tasks(timeout=5.0)
        # 强制把还在 _pending_compress 里的待压缩消息立刻压缩，避免丢在内存里
        await memory_manager.flush_pending()
        memory_manager.save()
        logger.info("[记忆] 已保存到磁盘")
    await cleanup_mcp()


async def bot(runner_args: RunnerArguments):
    """机器人主入口，兼容 Pipecat Cloud。

    Args:
        runner_args: Pipecat 运行器传来的配置参数。
    """
    from pipecat.runner.utils import create_transport
    from pipecat.transports.base_transport import TransportParams

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)
