#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat 独立语音助手启动脚本。

通过 .env 配置 STT/TTS/LLM 服务，启动一个支持语音对话的 Pipecat 机器人。
级联流水线：语音识别 → 大模型 → 语音合成

运行方式：

    uv run python bot_standalone.py

环境变量配置（.env）：

- MIMO_API_KEY: MiMo API 密钥
- PIPECAT_STT_PROVIDER: STT 服务商（mimo / whisper，默认 mimo）
- PIPECAT_STT_MODEL: Whisper 模型名（tiny / base / small / medium / large-v3）
- PIPECAT_TTS_PROVIDER: TTS 服务商（mimo / kokoro / piper / voxcpm，默认 mimo）
- PIPECAT_TTS_VOICE: TTS 音色
- PIPECAT_TTS_LANGUAGE: TTS 语言代码
- PIPECAT_VOXCPM_URL: VoxCPM 服务地址
- PIPECAT_LLM_BASE_URL: LLM API 地址（默认 https://token-plan-cn.xiaomimimo.com/v1）
- PIPECAT_LLM_MODEL: LLM 模型名（默认 mimo-v2.5）
- PIPECAT_LLM_API_KEY: LLM API 密钥（为空时回退到 MIMO_API_KEY）
- PIPECAT_WAKE_WORD: 唤醒词（逗号分隔，为空则始终唤醒）
- PIPECAT_WAKE_TIMEOUT: 唤醒超时时间（秒，默认 60）
"""

import os
import re
import random
import asyncio

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
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

import httpx
from datetime import datetime, timezone, timedelta
from html import unescape

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from pipecat_mcp_server.agent import _load_audio_data_uri
from pipecat_mcp_server.processors.mimo_stt import MiMoSTTService
from pipecat_mcp_server.processors.mimo_tts import MiMoTTSService
from pipecat_mcp_server.processors.voxcpm_tts import VoxCPMTTSService
from zhconv import convert as zh_convert

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------
DEFAULT_STT_PROVIDER = "mimo"
DEFAULT_STT_MODEL = "medium"
DEFAULT_TTS_PROVIDER = "mimo"
DEFAULT_MIMO_TTS_VOICE = "mimo_default"
DEFAULT_KOKORO_TTS_VOICE = "af_heart"
DEFAULT_PIPER_TTS_VOICE = "zh_CN-huayan-medium"
DEFAULT_VOXCPM_TTS_URL = "http://localhost:8000"
DEFAULT_VOXCPM_TTS_MODEL = "openbmb/VoxCPM2"
DEFAULT_VOXCPM_TTS_VOICE = "default"
DEFAULT_VOXCPM_TTS_SEED = 2028
DEFAULT_MIMO_TTS_LANGUAGE = "zh"
DEFAULT_KOKORO_TTS_LANGUAGE = "en"
DEFAULT_LLM_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_LLM_MODEL = "mimo-v2.5"
DEFAULT_MCP_SERVERS = ""

# 唤醒后随机问候语列表
GREETINGS = [
    "请吩咐。",
    "我在呢，请说。",
    "你好，有什么可以帮你的？",
    "说吧，我听着呢。",
    "嗯，在呢。",
]

# LLM 系统提示词
SYSTEM_PROMPT = (
    "你是一个友好的中文语音助手。请用简洁、自然的口语回答用户的问题。"
    "回答尽量控制在 2-3 句话内，不要使用 markdown 格式。"
    "你可以使用工具函数来获取实时信息。内置工具包括：搜索网络、获取当前时间。"
    "如果配置了 MCP 服务，你还会有浏览器工具可用（打开网页、截图、点击、输入等）。"
    "调用工具时请使用正确的工具名称。"
    "当用户询问需要最新信息的问题时，主动调用相应的工具。"
    "工具返回的网页内容如果太长，请提炼出用户最关心的要点进行回答。"
)

# ---------------------------------------------------------------------------
# 工具函数（LLM Function Calling）
# ---------------------------------------------------------------------------

_tz_cn = timezone(timedelta(hours=8))


async def tool_get_current_time(params: FunctionCallParams):
    """获取当前北京时间。"""
    now = datetime.now(_tz_cn)
    await params.result_callback({
        "datetime": now.strftime("%Y年%m月%d日 %H:%M:%S"),
        "weekday": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()],
    })


async def tool_search_web(params: FunctionCallParams):
    """搜索网页并返回结果摘要。

    Args:
        query: 搜索关键词。
    """
    query = params.arguments.get("query", "")
    if not query:
        await params.result_callback({"error": "未提供搜索关键词"})
        return

    logger.info(f"[工具] 网页搜索: {query}")
    try:
        import urllib.parse
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                search_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            resp.raise_for_status()
            html_text = resp.text

        import re
        results = []
        for m in re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html_text,
            re.DOTALL | re.IGNORECASE,
        ):
            link = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
            if title and link.startswith("http"):
                results.append({"title": unescape(title), "url": link, "snippet": unescape(snippet)})
            if len(results) >= 5:
                break

        if not results:
            await params.result_callback({"query": query, "results": [], "message": "未找到相关结果"})
        else:
            await params.result_callback({"query": query, "results": results})
    except Exception as e:
        logger.warning(f"[工具] 搜索失败: {e}")
        await params.result_callback({"error": f"搜索失败: {e}"})


# MCP 会话管理器（bot 停止时清理）
_mcp_sessions: list[ClientSession] = []
_mcp_transports: list = []
_mcp_http_clients: list[httpx.AsyncClient] = []


async def _load_mcp_tools(server_urls: list[str]) -> list:
    """连接 MCP 服务器并动态注册其提供的工具。

    通过 MCP Streamable HTTP 协议连接每个 MCP 服务器，
    调用 tools/list 获取可用工具列表，
    为每个工具生成一个 Pipecat FunctionCallParams 兼容的 wrapper 函数。

    Args:
        server_urls: MCP 服务器端点列表，例如 ['http://localhost:9091/mcp']。

    Returns:
        Pipecat DirectFunction 工具函数列表。
    """
    tools = []
    http_client = httpx.AsyncClient(trust_env=False, timeout=30.0)
    _mcp_http_clients.append(http_client)
    for url in server_urls:
        url = url.strip()
        if not url:
            continue
        logger.info(f"[MCP] 正在连接 MCP 服务: {url}")
        try:
            transport_ctx = streamable_http_client(url, http_client=http_client)
            read, write, _ = await transport_ctx.__aenter__()
            _mcp_transports.append(transport_ctx)

            session = ClientSession(read, write)
            await session.__aenter__()
            _mcp_sessions.append(session)

            await session.initialize()
            result = await session.list_tools()

            for tool in result.tools:
                # 避免与内置工具重名
                wrapper_name = f"mcp_{tool.name}" if tool.name in {"get_current_time", "browser_navigate", "browser_screenshot", "search_web"} else tool.name
                # 构建 docstring：description + 参数列表
                doc = tool.description or f"MCP 工具: {tool.name}"
                input_schema = tool.inputSchema or {}
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])
                if properties:
                    doc += "\n\nArgs:\n"
                    for pname, pinfo in properties.items():
                        ptype = pinfo.get("type", "any")
                        pdesc = pinfo.get("description", "")
                        req = " (必填)" if pname in required else ""
                        doc += f"    {pname}: {ptype}{req}。{pdesc}\n"

                # 用闭包创建 wrapper，捕获 tool.name 和 session
                async def _make_wrapper(_name: str, _session: ClientSession):
                    async def wrapper(params: FunctionCallParams):
                        logger.info(f"[MCP] 调用工具: {_name}({dict(params.arguments)})")
                        try:
                            call_result = await _session.call_tool(_name, dict(params.arguments))
                            # 提取 text content
                            texts = []
                            for block in call_result.content:
                                if hasattr(block, "text"):
                                    texts.append(block.text)
                            result_text = "\n".join(texts) if texts else str(call_result.content)
                            await params.result_callback({"result": result_text})
                        except Exception as e:
                            logger.warning(f"[MCP] 工具调用失败 {_name}: {e}")
                            await params.result_callback({"error": str(e)})
                    return wrapper

                wrapper = await _make_wrapper(tool.name, session)
                wrapper.__name__ = wrapper_name
                wrapper.__doc__ = doc
                wrapper.__qualname__ = wrapper_name

                tools.append(wrapper)
                logger.info(f"[MCP] 已注册工具: {wrapper_name} (来自 {url})")

            logger.info(f"[MCP] {url} 加载完成，共 {len(result.tools)} 个工具")
        except Exception as e:
            logger.warning(f"[MCP] 连接失败 {url}: {e}")

    return tools


async def _cleanup_mcp():
    """关闭所有 MCP 客户端会话。"""
    for session in _mcp_sessions:
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            pass
    _mcp_sessions.clear()
    for transport in _mcp_transports:
        try:
            await transport.__aexit__(None, None, None)
        except Exception:
            pass
    _mcp_transports.clear()
    for client in _mcp_http_clients:
        try:
            await client.aclose()
        except Exception:
            pass
    _mcp_http_clients.clear()


# 工具列表
BOT_TOOLS = [
    tool_get_current_time,
    tool_search_web,
]

_PUNCT_RE = re.compile(r"[]\s，。！？、；：""''（）《》【】,.!?;:\"'(){}[]+")


def _normalize(text: str) -> str:
    """移除标点和空白，用于唤醒词匹配。"""
    return _PUNCT_RE.sub("", text)


def _find_wake_word(text: str, wake_words: list[str]) -> str | None:
    """在文本中查找第一个匹配的唤醒词（忽略标点空白）。"""
    normalized = _normalize(text)
    for w in wake_words:
        if w in normalized:
            return w
    return None


def _create_stt_service():
    """根据环境变量创建语音识别服务。

    Returns:
        配置完成的 STT 服务实例。

    Raises:
        ValueError: 如果 STT provider 不支持。
    """
    provider = os.environ.get("PIPECAT_STT_PROVIDER", DEFAULT_STT_PROVIDER).strip().lower()
    if provider == "mimo":
        return MiMoSTTService(
            api_key=os.environ.get("MIMO_API_KEY"),
            language="zh",
        )

    if provider == "whisper":
        from pipecat.services.whisper.stt import WhisperSTTService
        from pipecat.transcriptions.language import Language

        return WhisperSTTService(
            settings=WhisperSTTService.Settings(
                model=os.environ.get("PIPECAT_STT_MODEL", DEFAULT_STT_MODEL),
                language=Language.ZH,
                no_speech_prob=float(os.environ.get("PIPECAT_STT_NO_SPEECH_PROB", "0.4")),
            ),
        )

    raise ValueError(f"不支持的 STT provider: {provider}")


def _create_tts_service():
    """根据环境变量创建语音合成服务。

    Returns:
        配置完成的 TTS 服务实例。

    Raises:
        ValueError: 如果 TTS provider 不支持。
    """
    provider = os.environ.get("PIPECAT_TTS_PROVIDER", DEFAULT_TTS_PROVIDER).strip().lower()
    if provider == "mimo":
        return MiMoTTSService(
            api_key=os.environ.get("MIMO_API_KEY"),
            voice=os.environ.get("PIPECAT_TTS_VOICE", DEFAULT_MIMO_TTS_VOICE),
            language=os.environ.get("PIPECAT_TTS_LANGUAGE", DEFAULT_MIMO_TTS_LANGUAGE),
        )

    if provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService
        from pipecat.transcriptions.language import Language

        lang_value = os.environ.get("PIPECAT_TTS_LANGUAGE", DEFAULT_KOKORO_TTS_LANGUAGE)
        language = Language(lang_value.strip().lower().replace("-", "_"))
        return KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice=os.environ.get("PIPECAT_TTS_VOICE", DEFAULT_KOKORO_TTS_VOICE),
                language=language,
            ),
        )

    if provider == "piper":
        from pipecat.services.piper.tts import PiperTTSService

        return PiperTTSService(
            settings=PiperTTSService.Settings(
                voice=os.environ.get("PIPECAT_TTS_VOICE", DEFAULT_PIPER_TTS_VOICE),
            ),
        )

    if provider == "voxcpm":
        return VoxCPMTTSService(
            base_url=os.environ.get("PIPECAT_VOXCPM_URL", DEFAULT_VOXCPM_TTS_URL),
            model=DEFAULT_VOXCPM_TTS_MODEL,
            voice=DEFAULT_VOXCPM_TTS_VOICE,
            ref_audio=_load_audio_data_uri(r"voice_samples/core-capability-1.wav"),
            ref_text="街口那个老周啊，媳妇走得早，一个人拉扯俩娃，白天蹬三轮，晚上还去夜市摆摊修鞋。现在俩孩子都有出息喽，想接他去城里享福——他不去，就守着那间小铺子。哎，人哪，骨头硬，心里头就踏实。",
            seed=DEFAULT_VOXCPM_TTS_SEED,
        )

    raise ValueError(f"不支持的 TTS provider: {provider}")


def _create_llm_service():
    """根据环境变量创建大模型服务。

    使用 OpenAI 兼容 API 接口，默认为 MiMo 模型。

    Returns:
        配置完成的 LLM 服务实例。
    """
    api_key = os.environ.get("PIPECAT_LLM_API_KEY") or os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise ValueError("需要设置 MIMO_API_KEY 或 PIPECAT_LLM_API_KEY 环境变量")

    base_url = os.environ.get("PIPECAT_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    model = os.environ.get("PIPECAT_LLM_MODEL", DEFAULT_LLM_MODEL)

    return OpenAILLMService(
        api_key=api_key,
        base_url=base_url,
        settings=OpenAILLMService.Settings(
            model=model,
            system_instruction=SYSTEM_PROMPT,
        ),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """组装并运行机器人的主逻辑。

    Args:
        transport: Pipecat 传输实例。
        runner_args: Runner 配置参数。
    """
    logger.info("正在启动独立语音助手...")

    # 唤醒词配置
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

    # 加载 MCP 工具
    mcp_server_urls = [
        u.strip() for u in os.environ.get("PIPECAT_MCP_SERVERS", DEFAULT_MCP_SERVERS).split(",") if u.strip()
    ]
    mcp_tools = await _load_mcp_tools(mcp_server_urls)
    all_tools = BOT_TOOLS + mcp_tools
    logger.info(f"总共注册 {len(all_tools)} 个工具（内置 {len(BOT_TOOLS)} + MCP {len(mcp_tools)}）")

    # 创建服务
    stt = _create_stt_service()
    tts = _create_tts_service()
    llm = _create_llm_service()

    context = LLMContext(tools=all_tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("客户端已连接")
        context.add_message(
            {"role": "developer", "content": "请向用户做一个简短的自我介绍，告诉他你是一个中文语音助手。"}
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
            matched = _find_wake_word(text, wake_words)
            if matched:
                logger.info(f"唤醒词 '{matched}' 检测到: '{text}'")
                awake = True
                await schedule_awake_timeout()
                remaining = _normalize(text).replace(matched, "", 1).strip()
                if remaining:
                    # 唤醒词 + 问题：更新最后一条用户消息为剩余内容
                    if context.messages:
                        last = context.messages[-1]
                        if last.get("role") == "user":
                            last["content"] = remaining
                else:
                    # 仅唤醒词：发送随机问候
                    greeting = random.choice(GREETINGS)
                    logger.info(f"播放唤醒问候: '{greeting}'")
                    context.add_message(
                        {"role": "developer", "content": f"请用语音说出: {greeting}"}
                    )
                    await worker.queue_frames([LLMRunFrame()])
            else:
                logger.debug(f"睡眠中忽略语音: '{text}'")
                # 移除最后一条用户消息，避免 LLM 响应
                if context.messages and context.messages[-1].get("role") == "user":
                    context.messages.pop()
        else:
            # 唤醒状态：正常处理
            if wake_words:
                cancel_awake_timeout()
                await schedule_awake_timeout()

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}助手: {message.content}"
        logger.info(f"转录: {line}")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()

    # 清理
    cancel_awake_timeout()
    await _cleanup_mcp()


async def bot(runner_args: RunnerArguments):
    """机器人主入口，兼容 Pipecat Cloud。

    Args:
        runner_args: Pipecat 运行器传来的配置参数。
    """
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
