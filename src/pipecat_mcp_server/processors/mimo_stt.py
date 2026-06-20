"""小米 MiMo-V2.5-ASR 语音识别服务（pipecat 自定义 STT）。

通过 OpenAI 兼容接口调用小米 MiMo 云端 ASR，中文及方言识别效果优于本地 Whisper。
批处理模式：在 VAD 检测到用户说完一段话后，把整段 WAV 音频 base64 编码后发给 API。
"""

import base64
import os
from collections.abc import AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5-asr"


class MiMoSTTService(SegmentedSTTService):
    """基于小米 MiMo-V2.5-ASR 的批处理语音识别服务。

    SegmentedSTTService 会在 VAD 检测到用户说完一段话后，把缓冲好的整段音频
    转成 WAV bytes 传给 run_stt()。这里只负责 base64 编码 + 调用 API + 返回文本。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = MIMO_DEFAULT_MODEL,
        language: str = "zh",
        base_url: str = MIMO_BASE_URL,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not self._api_key:
            raise ValueError(
                "MiMoSTTService 需要提供 api_key，或设置环境变量 MIMO_API_KEY"
            )
        self._model = model
        self._language = language
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """把一段 WAV 音频发给 MiMo ASR，返回识别文本。

        Args:
            audio: 完整的 WAV 文件 bytes（含 RIFF 头，16-bit PCM mono），
                   由 SegmentedSTTService 在 VAD 停顿后拼装好。
        """
        if not audio:
            return

        await self.start_processing_metrics()

        try:
            audio_b64 = base64.b64encode(audio).decode("utf-8")
            data_url = f"data:audio/wav;base64,{audio_b64}"

            extra_body = {"asr_options": {"language": self._language}}

            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": data_url},
                            }
                        ],
                    }
                ],
                extra_body=extra_body,
            )

            text = ""
            if completion.choices:
                text = (completion.choices[0].message.content or "").strip()

            language = Language.ZH if self._language == "zh" else Language.EN

            if text:
                logger.debug(f"MiMo ASR transcription: [{text}]")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                )
        except Exception as e:
            logger.exception(f"MiMo ASR 调用失败: {e}")
            yield ErrorFrame(error=f"MiMo ASR 调用失败: {e}")
        finally:
            await self.stop_processing_metrics()
