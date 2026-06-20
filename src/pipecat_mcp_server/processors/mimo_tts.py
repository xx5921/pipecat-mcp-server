"""小米 MiMo-V2.5-TTS 语音合成服务（pipecat 自定义 TTS）。

通过 OpenAI 兼容的 chat/completions 接口（stream=True）调用小米 MiMo 云端 TTS，
使用 PCM16 格式流式返回 24kHz 原始音频。
"""

import base64
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5-tts"
MIMO_DEFAULT_VOICE = "mimo_default"

# MiMo PCM16 输出采样率
MIMO_PCM16_SAMPLE_RATE = 24000

# MiMo 预置中文音色
PRESET_VOICES = [
    "mimo_default",
    "冰糖",
    "茉莉",
    "苏打",
    "白桦",
    "Mia",
    "Chloe",
    "Milo",
    "Dean",
]


@dataclass
class MiMoTTSSettings(TTSSettings):
    """MiMo TTS 服务配置。"""

    voice: str | None = None
    speed: float | None = None


class MiMoTTSService(TTSService):
    """基于小米 MiMo-V2.5-TTS 的流式语音合成服务。

    通过 OpenAI 兼容的 chat/completions（stream=True, audio.format=pcm16）调用。
    文本以 role=assistant 消息发送，音频通过 SSE 流式返回裸 PCM16 字节。
    """

    Settings = MiMoTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = MIMO_DEFAULT_MODEL,
        voice: str = MIMO_DEFAULT_VOICE,
        language: str = "zh",
        base_url: str = MIMO_BASE_URL,
        audio_format: str = "pcm16",
        speed: float | None = None,
        sample_rate: int = MIMO_PCM16_SAMPLE_RATE,
        settings: Settings | None = None,
        **kwargs,
    ):
        """初始化 MiMo TTS 服务。

        Args:
            api_key: MiMo API 密钥，默认从环境变量 MIMO_API_KEY 读取。
            model: TTS 模型名，默认 mimo-v2.5-tts。
            voice: 预置音色名，可选 mimo_default / 冰糖 / 茉莉 / 苏打 / 白桦 等。
            language: 语言代码，默认 zh。
            base_url: API 地址。
            audio_format: 音频格式，默认 pcm16（24kHz, 16-bit, mono, little-endian）。
            speed: 语速（当前 MiMo 端可能不生效，预留）。
            sample_rate: 输出采样率，默认 24000。
            settings: 运行时可更新的设置。
        """
        default_settings = self.Settings(
            model=model,
            voice=voice,
            language=language,
            speed=speed,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not self._api_key:
            raise ValueError(
                "MiMoTTSService 需要提供 api_key，或设置环境变量 MIMO_API_KEY"
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._audio_format = audio_format
        self._tts_sample_rate = sample_rate
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            timeout=60.0,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        if "voice" in changed:
            self._settings.voice = changed["voice"]
        if "speed" in changed:
            self._settings.speed = changed["speed"]
        if "model" in changed:
            self._model = changed["model"]
        if "language" in changed:
            self._settings.language = changed["language"]
        return changed

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """流式合成语音。

        使用 MiMo chat/completions（stream=True, audio.format=pcm16），
        从 SSE 事件中提取 base64 编码的 PCM16 音频块，解码后
        通过 _stream_audio_frames_from_iterator 输出 TTSAudioRawFrame。

        Args:
            text: 要合成的文本。
            context_id: 当前 TTS 上下文 ID。
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        _voice = assert_given(self._settings.voice)
        if _voice is None:
            _voice = MIMO_DEFAULT_VOICE

        audio_config: dict[str, Any] = {
            "voice": _voice,
            "format": self._audio_format,
        }
        if self._settings.speed is not None:
            audio_config["speed"] = self._settings.speed

        try:
            await self.start_tts_usage_metrics(text)

            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "assistant", "content": text}],
                audio=audio_config,
                modalities=["text", "audio"],
                stream=True,
            )

            async def audio_chunk_iterator():
                async for chunk in completion:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    audio = getattr(delta, "audio", None)
                    if audio is not None:
                        assert isinstance(audio, dict), (
                            f"Expected audio to be a dict, got {type(audio)}"
                        )
                        data_b64 = audio.get("data", "")
                        if data_b64:
                            yield base64.b64decode(data_b64)

            async for frame in self._stream_audio_frames_from_iterator(
                audio_chunk_iterator(),
                in_sample_rate=self._tts_sample_rate,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(error=f"MiMo TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
            logger.debug(f"{self}: Finished TTS [{text}]")
