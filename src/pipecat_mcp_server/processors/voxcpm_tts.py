"""VoxCPM TTS service implementation using nanovllm-voxcpm HTTP API.

通过 nanovllm-voxcpm 提供的 OpenAI 兼容 `/v1/audio/speech` 接口实现
文本转语音功能。服务端以 `response_format=pcm` 直接返回 48 kHz s16le
裸 PCM 流，客户端使用 httpx 异步流式接收并通过 pipecat 的
`_stream_audio_frames_from_iterator` 切片为 TTSAudioRawFrame。
"""

import os
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# VoxCPM 默认配置（对齐官方 48 kHz s16le 标准）
VOXCPM_DEFAULT_URL = "http://localhost:8000"
VOXCPM_DEFAULT_MODEL = "openbmb/VoxCPM2"
VOXCPM_DEFAULT_VOICE = "default"
VOXCPM_DEFAULT_SAMPLE_RATE = 48000  # VoxCPM 服务端 PCM 输出固定为 48 kHz
VOXCPM_DEFAULT_SEED = 2028
VOXCPM_ENDPOINT = "/v1/audio/speech"


@dataclass
class VoxCPMTTSSettings(TTSSettings):
    """VoxCPM TTS 服务配置。"""

    voice: str | None = None
    seed: int | None = None


class VoxCPMTTSService(TTSService):
    """基于 nanovllm-voxcpm 的流式语音合成服务。

    通过 OpenAI 兼容的 `/v1/audio/speech` 接口调用 VoxCPM 服务，
    以 `response_format=pcm` 接收 48 kHz s16le 裸 PCM 字节流，并
    使用 pipecat 的流式切片工具输出 `TTSAudioRawFrame`。

    需要先部署 nanovllm-voxcpm 服务。
    """

    Settings = VoxCPMTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str = VOXCPM_DEFAULT_URL,
        model: str = VOXCPM_DEFAULT_MODEL,
        voice: str = VOXCPM_DEFAULT_VOICE,
        seed: int = VOXCPM_DEFAULT_SEED,
        sample_rate: int = VOXCPM_DEFAULT_SAMPLE_RATE,
        timeout: float = 60.0,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """初始化 VoxCPM TTS 服务。

        Args:
            base_url: VoxCPM 服务地址，例如 ``http://100.84.59.58:8100``。
            model: 模型名称，默认 ``openbmb/VoxCPM2``。
            voice: 预置音色名，默认 ``default``。
            seed: 音色随机种子，固定种子可复现同一音色。
            sample_rate: 服务端 PCM 输出采样率，默认 48000。
            timeout: HTTP 请求超时时间（秒）。
            ref_audio: 参考音频的 base64 data URI，用于自定义音色克隆。
            ref_text: 与 ``ref_audio`` 配套的参考文本。
            settings: 运行时可更新的设置。
            **kwargs: 传递给父类 TTSService 的额外参数。
        """
        default_settings = self.Settings(voice=voice, seed=seed, model=model, language=None)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._voxcpm_sample_rate = sample_rate
        self._client = httpx.AsyncClient(timeout=timeout)

    def can_generate_metrics(self) -> bool:
        """指示此服务支持 TTFB 和使用指标。"""
        return True

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """更新服务设置。

        Args:
            delta: 要更新的设置。

        Returns:
            实际发生变化的设置字典。
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        if "voice" in changed:
            self._settings.voice = changed["voice"]
        if "seed" in changed:
            self._settings.seed = changed["seed"]
        return changed

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """流式合成语音。

        通过 OpenAI 兼容的 `/v1/audio/speech` 接口请求 VoxCPM，
        以 ``response_format=pcm`` 接收 48 kHz s16le 裸 PCM 字节流，
        边收边切分为 ``TTSAudioRawFrame``。

        Args:
            text: 要合成的文本。
            context_id: 当前 TTS 上下文 ID。
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        payload: dict[str, Any] = {
            "model": self._model,
            "input": text,
            "voice": self._settings.voice or VOXCPM_DEFAULT_VOICE,
            "stream": True,
            "response_format": "pcm",
        }
        if self._settings.seed is not None:
            payload["seed"] = self._settings.seed
        if self._ref_audio:
            payload["ref_audio"] = self._ref_audio
        if self._ref_text:
            payload["ref_text"] = self._ref_text

        try:
            await self.start_tts_usage_metrics(text)

            async with self._client.stream(
                "POST",
                f"{self._base_url}{VOXCPM_ENDPOINT}",
                json=payload,
                timeout=self._timeout,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise httpx.HTTPStatusError(
                        f"VoxCPM HTTP {response.status_code}: {body.decode('utf-8', 'ignore')}",
                        request=response.request,
                        response=response,
                    )

                async def pcm_chunk_iterator() -> AsyncIterator[bytes]:
                    """逐块读取 VoxCPM 返回的裸 PCM 字节流。"""
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk

                async for frame in self._stream_audio_frames_from_iterator(
                    pcm_chunk_iterator(),
                    in_sample_rate=self._voxcpm_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame

        except httpx.HTTPStatusError as e:
            logger.error(f"{self} HTTP error: {e}")
            yield ErrorFrame(error=f"VoxCPM HTTP error: {e}")
        except httpx.RequestError as e:
            logger.error(f"{self} request error: {e}")
            yield ErrorFrame(error=f"VoxCPM request error: {e}")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(error=f"VoxCPM TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
            logger.debug(f"{self}: Finished TTS [{text}]")

    async def stop(self):
        """停止服务并清理资源。"""
        await self._client.aclose()
        await super().stop()
