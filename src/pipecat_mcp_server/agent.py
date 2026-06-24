#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat MCP Agent for voice I/O over MCP protocol.

This module provides the `PipecatMCPAgent` class that exposes voice input/output
capabilities through MCP tools. It manages a Pipecat pipeline with STT and TTS
services, allowing an MCP client to listen for user speech and speak responses.
"""

import asyncio
import base64
import os
import random
import re
from pathlib import Path
from typing import Any, Optional

import pipecat.processors.frameworks.rtvi.models as RTVI
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import (
    DailyRunnerArguments,
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_mcp_server.processors.mimo_stt import MiMoSTTService
from pipecat_mcp_server.processors.mimo_tts import MiMoTTSService
from pipecat_mcp_server.processors.screen_capture import ScreenCaptureProcessor
from pipecat_mcp_server.processors.vision import VisionProcessor
from pipecat_mcp_server.processors.voxcpm_tts import VoxCPMTTSService
from zhconv import convert as zh_convert

load_dotenv(override=True)

GREETINGS = [
    "请吩咐。",
    "我在呢，请说。",
    "你好，有什么可以帮你的？",
    "说吧，我听着呢。",
    "嗯，在呢。",
]

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
# 默认音色克隆参考音频（模块同级 voice/girl_voice.wav）
DEFAULT_VOXCPM_REF_AUDIO = str(Path(__file__).parent / "voice" / "girl_voice.wav")
DEFAULT_VOXCPM_REF_TEXT = "你好啊，今天是开心的一天"
DEFAULT_MIMO_TTS_LANGUAGE = "zh"
DEFAULT_KOKORO_TTS_LANGUAGE = "en"


_PUNCT_RE = re.compile(r"[]\s，。！？、；：""''（）《》【】,.!?;:\"'(){}[]+")

def _normalize(text: str) -> str:
    """Remove punctuation and whitespace for wake word matching."""
    return _PUNCT_RE.sub("", text)


def _find_wake_word(text: str, wake_words: list[str]) -> str | None:
    """Return the first wake word found in text, or None.

    Punctuation and whitespace in the transcribed text are ignored
    during matching.
    """
    normalized = _normalize(text)
    for w in wake_words:
        if w in normalized:
            return w
    return None


def _resolve_language(value: str) -> Language:
    """将环境变量中的语言代码转换为 Pipecat 语言枚举.

    Args:
        value: 环境变量中的语言代码，例如 zh、en、ja。

    Returns:
        Pipecat 的 Language 枚举值。

    """
    language = value.strip().lower().replace("-", "_")
    return Language(language)


def _load_audio_data_uri(path: str) -> str | None:
    """读取本地音频文件并转换为 base64 data URI。

    用于 VoxCPM 音色克隆（ref_audio）。返回结果可直接作为
    OpenAI 兼容 `/v1/audio/speech` 请求体中的 `ref_audio` 字段值。

    Args:
        path: 本地音频文件路径，支持 .wav 等。

    Returns:
        形如 ``data:audio/wav;base64,<base64>`` 的字符串；若文件
        不存在则返回 ``None``。

    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"没有找到参考音频文件: {path}，请先放一个短音频在同目录下。")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


class PipecatMCPAgent:
    """Pipecat MCP Agent that exposes voice I/O tools.

    Tools:
    - listen(): Wait for user speech and return transcription
    - speak(text): Speak text to the user via TTS
    """

    # Sentinel value to indicate client disconnection
    _DISCONNECT_SENTINEL = object()

    def __init__(
        self,
        transport: BaseTransport,
        runner_args: RunnerArguments,
    ):
        """Initialize the Pipecat MCP Agent.

        Args:
            transport: Transport for audio I/O (Daily, Twilio, or WebRTC).
            runner_args: Runner configuration arguments.

        """
        self._transport = transport
        self._runner_args = runner_args

        self._task: Optional[asyncio.Task] = None
        self._pipeline_task: Optional[PipelineTask] = None
        self._pipeline_runner: Optional[PipelineRunner] = None
        self._assistant_aggregator: Optional[Any] = None
        self._user_speech_queue: asyncio.Queue[Any] = asyncio.Queue()

        # Wake word state
        raw = os.environ.get("PIPECAT_WAKE_WORD", "")
        self._wake_words: list[str] = [w.strip() for w in raw.split(",") if w.strip()]
        self._awake: bool = True  # start awake so initial conversation flows naturally
        self._awake_timeout_secs: float = float(os.environ.get("PIPECAT_WAKE_TIMEOUT", "30"))
        self._awake_timeout_task: Optional[asyncio.Task] = None

        self._started = False

    async def start(self):
        """Start the voice pipeline.

        Initializes STT and TTS services, creates the processing pipeline,
        and starts it in the background. The pipeline remains active until
        `stop()` is called.

        Raises:
            ValueError: If required API keys are missing from environment.

        """
        if self._started:
            return

        logger.info("Starting Pipecat MCP Agent pipeline...")

        # Create services
        stt = self._create_stt_service()
        tts = self._create_tts_service()

        context = LLMContext()
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())
                    ]
                ),
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            ),
        )

        # Store assistant aggregator so speak() can push text directly to it
        # for display in the WebRTC client UI (TTS consumes text frames in the
        # pipeline, so they never reach the assistant aggregator downstream).
        self._assistant_aggregator = assistant_aggregator

        self._screen_capture = ScreenCaptureProcessor()
        self._vision = VisionProcessor()

        # Create pipeline with parallel branches:
        # - Main branch: audio processing (STT → aggregator → TTS)
        # - Vision branch: saves frames to disk on demand
        pipeline = Pipeline(
            [
                self._transport.input(),
                self._screen_capture,
                ParallelPipeline(
                    [stt, user_aggregator, tts],
                    [self._vision],
                ),
                # Assistant aggregator before the transport, because we want to
                # keep everyting from the client.
                assistant_aggregator,
                self._transport.output(),
            ]
        )

        self._pipeline_task = PipelineTask(
            pipeline,
            cancel_on_idle_timeout=False,
        )

        self._pipeline_runner = PipelineRunner(handle_sigterm=True)

        @self._transport.event_handler("on_client_connected")
        async def on_connected(transport, client):
            logger.info(f"Client connected")

        @self._transport.event_handler("on_client_disconnected")
        async def on_disconnected(transport, client):
            logger.info(f"Client disconnected")
            if not self._pipeline_task:
                return

            if isinstance(self._runner_args, DailyRunnerArguments):
                await self._user_speech_queue.put("I just disconnected, but I might come back.")
            else:
                await self._user_speech_queue.put(self._DISCONNECT_SENTINEL)
                await self._pipeline_task.cancel()

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
            if not message.content:
                return

            text = zh_convert(message.content, "zh-cn")

            if not self._awake and self._wake_words:
                # Sleeping: check for wake word
                matched = _find_wake_word(text, self._wake_words)
                if matched:
                    logger.info(f"Wake word '{matched}' detected in: '{text}'")
                    self._awake = True
                    self._schedule_awake_timeout()
                    remaining = _normalize(text).replace(matched, "", 1).strip()
                    if remaining:
                        # Wake word + question: forward question, skip greeting
                        await self._user_speech_queue.put(remaining)
                    else:
                        # Wake word alone: speak a random greeting
                        greeting = random.choice(GREETINGS)
                        logger.info(f"Speaking wake greeting: '{greeting}'")
                        await self.speak(greeting)
                else:
                    logger.debug(f"Ignoring speech while asleep: '{text}'")
            else:
                # Awake: process speech normally
                await self._user_speech_queue.put(text)
                if self._wake_words:
                    self._cancel_awake_timeout()
                    self._schedule_awake_timeout()

        # Start pipeline in background
        self._task = asyncio.create_task(self._pipeline_runner.run(self._pipeline_task))

        # Start awake timeout so the initial greeting/conversation flows
        # naturally; after inactivity the agent returns to sleep.
        if self._wake_words:
            self._schedule_awake_timeout()

        self._started = True
        logger.info("Pipecat MCP Agent started!")

    async def stop(self):
        """Stop the voice pipeline.

        Sends an `EndFrame` to gracefully shut down the pipeline and waits
        for the background task to complete.
        """
        if not self._started:
            return

        logger.info("Stopping Pipecat MCP agent...")

        if self._awake_timeout_task:
            self._awake_timeout_task.cancel()
            self._awake_timeout_task = None

        if self._pipeline_task:
            await self._pipeline_task.queue_frame(EndFrame())

        if self._task:
            await self._task

        self._started = False
        logger.info("Pipecat MCP Agent stopped")

    async def listen(self) -> str:
        """Wait for user speech and return the transcribed text.

        Blocks until the user completes an utterance (detected via VAD).
        Starts the pipeline automatically if not already running.

        Returns:
            The transcribed text from the user's speech.

        Raises:
            RuntimeError: If the pipeline task is not initialized.

        """
        if not self._started:
            await self.start()

        if not self._pipeline_task:
            raise RuntimeError("Pipecat MCP Agent not initialized")

        text = await self._user_speech_queue.get()

        # Check if this is a disconnect signal
        if text is self._DISCONNECT_SENTINEL:
            raise RuntimeError("I just disconnected, but I might come back.")

        return text

    async def speak(self, text: str):
        """Speak text to the user using text-to-speech.

        Queues LLM response frames to synthesize and play the given text.
        Starts the pipeline automatically if not already running.

        Args:
            text: The text to speak to the user.

        Raises:
            RuntimeError: If the pipeline task is not initialized.

        """
        if not self._started:
            await self.start()

        if not self._pipeline_task:
            raise RuntimeError("Pipecat MCP Agent not initialized")

        # Push text directly to the WebRTC data channel via transport message,
        # so it appears in the client UI. The pipeline path (below) only produces
        # audio because TTS consumes LLMTextFrame and outputs TTSAudioRawFrame.
        bot_output = RTVI.BotOutputMessage(
            data=RTVI.BotOutputMessageData(
                text=text,
                aggregated_by="sentence",
                spoken=True,
            )
        )
        await self._transport.output().queue_frame(
            OutputTransportMessageUrgentFrame(message=bot_output.model_dump())
        )

        # Pipeline path: TTS converts text to audio for the user to hear
        await self._pipeline_task.queue_frames(
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame(text=text),
                LLMFullResponseEndFrame(),
            ]
        )

        # Reset awake timeout so the user has a full window to respond
        # after the bot finishes speaking, not from their last utterance.
        if self._awake and self._wake_words:
            self._cancel_awake_timeout()
            self._schedule_awake_timeout()

    async def list_windows(self) -> list[dict]:
        """List all open windows via the screen capture backend.

        Returns:
            A list of dicts with title, app_name, and window_id fields.

        """
        windows = await self._screen_capture._backend.list_windows()
        return [
            {"title": w.title, "app_name": w.app_name, "window_id": w.window_id} for w in windows
        ]

    async def screen_capture(self, window_id: Optional[int] = None) -> Optional[int]:
        """Switch screen capture to a different window or full screen.

        Args:
            window_id: Window ID to capture (from list_windows()), or None for full screen.

        Returns:
            The window ID if found, or None if the window was not found or capturing full screen.

        """
        return await self._screen_capture.screen_capture(window_id)

    async def capture_screenshot(self) -> str:
        """Capture a screenshot from the current screen capture stream.

        Saves the next frame to a temporary PNG file. Screen capture
        must already be started via screen_capture().

        Returns:
            The absolute path to the saved image file.

        """
        self._vision.request_capture()
        return await self._vision.get_result()

    @staticmethod
    def _create_stt_service() -> STTService:
        """根据环境变量创建语音识别服务.

        Returns:
            配置完成的 STT 服务实例。

        """
        provider = os.environ.get("PIPECAT_STT_PROVIDER", DEFAULT_STT_PROVIDER).strip().lower()
        if provider == "mimo":
            return MiMoSTTService(
                api_key=os.environ.get("MIMO_API_KEY"),
                language="zh",
            )

        if provider == "whisper":
            return WhisperSTTService(
                settings=WhisperSTTService.Settings(
                    model=os.environ.get("PIPECAT_STT_MODEL", DEFAULT_STT_MODEL),
                    language=Language.ZH,
                    no_speech_prob=float(os.environ.get("PIPECAT_STT_NO_SPEECH_PROB", "0.4")),
                ),
            )

        raise ValueError(f"Unsupported STT provider: {provider}")

    @staticmethod
    def _create_tts_service() -> TTSService:
        """根据环境变量创建语音合成服务.

        Returns:
            配置完成的 TTS 服务实例。

        """
        provider = os.environ.get("PIPECAT_TTS_PROVIDER", DEFAULT_TTS_PROVIDER).strip().lower()
        if provider == "mimo":
            return MiMoTTSService(
                api_key=os.environ.get("MIMO_API_KEY"),
                voice=os.environ.get("PIPECAT_TTS_VOICE", DEFAULT_MIMO_TTS_VOICE),
                language=os.environ.get("PIPECAT_TTS_LANGUAGE", DEFAULT_MIMO_TTS_LANGUAGE),
            )

        if provider == "kokoro":
            return KokoroTTSService(
                settings=KokoroTTSService.Settings(
                    voice=os.environ.get("PIPECAT_TTS_VOICE", DEFAULT_KOKORO_TTS_VOICE),
                    language=_resolve_language(
                        os.environ.get("PIPECAT_TTS_LANGUAGE", DEFAULT_KOKORO_TTS_LANGUAGE)
                    ),
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
                ref_audio=_load_audio_data_uri(DEFAULT_VOXCPM_REF_AUDIO),
                ref_text=DEFAULT_VOXCPM_REF_TEXT,
            )

        raise ValueError(f"Unsupported TTS provider: {provider}")

    def set_wake_word(self, word: str) -> None:
        """Set the wake word(s) and enter sleep mode.

        Args:
            word: Comma-separated wake word phrases (e.g. "小林小林,小琳小琳").
                Empty string disables wake word.

        """
        word = word.strip()
        self._wake_words = [w.strip() for w in word.split(",") if w.strip()]
        if self._wake_words:
            self._awake = False
            logger.info(f"Wake words set to {self._wake_words}. Agent is now asleep.")
        else:
            self._awake = True
            self._cancel_awake_timeout()
            logger.info("Wake words cleared. Agent is always awake.")

    def get_wake_word(self) -> dict:
        """Get the current wake word configuration.

        Returns:
            A dict with 'words' and 'awake' keys.

        """
        return {
            "words": self._wake_words,
            "awake": self._awake,
        }

    def _schedule_awake_timeout(self):
        """Schedule a task to return the agent to sleep after the timeout."""
        if self._awake_timeout_task:
            self._awake_timeout_task.cancel()
        self._awake_timeout_task = asyncio.create_task(self._awake_timeout())

    def _cancel_awake_timeout(self):
        """Cancel the pending awake timeout task."""
        if self._awake_timeout_task:
            self._awake_timeout_task.cancel()
            self._awake_timeout_task = None

    async def _awake_timeout(self):
        """Wait for the timeout, then return agent to sleep."""
        try:
            await asyncio.sleep(self._awake_timeout_secs)
            logger.info(f"Wake timeout ({self._awake_timeout_secs}s) expired, returning to sleep")
            self._awake = False
        except asyncio.CancelledError:
            pass  # Timeout was replaced by a new utterance


async def create_agent(runner_args: RunnerArguments) -> PipecatMCPAgent:
    """Create a PipecatMCPAgent with the appropriate transport.

    Args:
        runner_args: Runner configuration specifying transport type and settings.

    Returns:
        A configured `PipecatMCPAgent` instance ready to be started.

    """
    transport_params = {}

    # Create transport based on runner args type
    if isinstance(runner_args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyParams

        transport_params["daily"] = lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            audio_in_filter=RNNoiseFilter(),
        )
    elif isinstance(runner_args, SmallWebRTCRunnerArguments):
        transport_params["webrtc"] = lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            audio_in_filter=RNNoiseFilter(),
        )
    elif isinstance(runner_args, WebSocketRunnerArguments):
        params_callback = lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_filter=RNNoiseFilter(),
        )
        transport_params["twilio"] = params_callback
        transport_params["telnyx"] = params_callback
        transport_params["plivo"] = params_callback
        transport_params["exotel"] = params_callback

    transport = await create_transport(runner_args, transport_params)
    return PipecatMCPAgent(transport, runner_args)
