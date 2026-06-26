"""Qwen3-ASR 语音识别服务（pipecat 自定义 STT）。

通过 vLLM 自部署的 Qwen3-ASR 服务，OpenAI Whisper 兼容 API。
批处理模式：在 VAD 检测到用户说完一段话后，把整段 WAV 音频发给 /v1/audio/transcriptions。
"""

import io
import os
import re
from collections.abc import AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

QWEN_DEFAULT_BASE_URL = "http://100.84.59.58:8200/v1"
QWEN_DEFAULT_MODEL = "qwen3-asr"

# Qwen3-ASR 返回的文本会带语言标签前缀，如 "language Chinese<asr_text>实际内容"
# 这个正则用于去除前缀，避免污染下游 LLM context 和记忆模块
_ASR_LANG_TAG_RE = re.compile(r"language\s+\w+<asr_text>\s*")


class QwenSTTService(SegmentedSTTService):
    """基于 vLLM 自部署 Qwen3-ASR 的批处理语音识别服务。

    SegmentedSTTService 会在 VAD 检测到用户说完一段话后，把缓冲好的整段音频
    转成 WAV bytes 传给 run_stt()。这里负责把 WAV 发给 Qwen3-ASR 的
    /v1/audio/transcriptions 接口并返回文本。
    """

    def __init__(
        self,
        *,
        api_key: str = "not-needed",
        model: str = QWEN_DEFAULT_MODEL,
        language: str = "zh",
        base_url: str = QWEN_DEFAULT_BASE_URL,
        timeout: float = 30.0,
        ttfs_p99_latency: float = 2.0,
        **kwargs,
    ):
        """初始化 Qwen3-ASR 服务。

        Args:
            api_key: API 密钥（自部署服务无需真实密钥）。
            model: 模型名称，默认 Qwen/Qwen3-ASR-1.7B。
            language: 识别语言代码，默认 zh。
            base_url: Qwen3-ASR 服务地址。
            timeout: 请求超时时间。
            ttfs_p99_latency: P99 语音结束到最终转写延迟（秒）。
        """
        settings = STTSettings(model=model, language=language)
        super().__init__(settings=settings, ttfs_p99_latency=ttfs_p99_latency, **kwargs)
        self._model = model
        self._language = language
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """把一段 WAV 音频发给 Qwen3-ASR，返回识别文本。

        Args:
            audio: 完整的 WAV 文件 bytes（含 RIFF 头，16-bit PCM mono），
                   由 SegmentedSTTService 在 VAD 停顿后拼装好。
        """
        if not audio:
            return

        await self.start_processing_metrics()

        try:
            wav_buffer = io.BytesIO(audio)
            wav_buffer.name = "audio.wav"

            transcription = await self._client.audio.transcriptions.create(
                model=self._model,
                file=wav_buffer,
                language=self._language,
            )

            text = (transcription.text or "").strip()
            # Qwen3-ASR 会加语言标签前缀（如 "language Chinese<asr_text>"），需去除
            text = _ASR_LANG_TAG_RE.sub("", text).strip()
            lang = Language.ZH if self._language == "zh" else Language.EN

            if text:
                logger.debug(f"Qwen3-ASR transcription: [{text}]")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    lang,
                )
        except Exception as e:
            logger.exception(f"Qwen3-ASR 调用失败: {e}")
            yield ErrorFrame(error=f"Qwen3-ASR 调用失败: {e}")
        finally:
            await self.stop_processing_metrics()
