"""STT/TTS/LLM 服务工厂函数。

根据环境变量创建 Pipecat 服务实例，独立于 MCP agent 路径。
"""

import os

from pipecat.services.openai.llm import OpenAILLMService

from pipecat_mcp_server.agent import _load_audio_data_uri
from pipecat_mcp_server.processors.mimo_stt import MiMoSTTService
from pipecat_mcp_server.processors.mimo_tts import MiMoTTSService
from pipecat_mcp_server.processors.qwen_stt import QwenSTTService
from pipecat_mcp_server.processors.voxcpm_tts import VoxCPMTTSService

from . import constants as C


def create_stt_service():
    """根据环境变量创建语音识别服务。

    Returns:
        配置完成的 STT 服务实例。

    Raises:
        ValueError: 如果 STT provider 不支持。
    """
    provider = os.environ.get("PIPECAT_STT_PROVIDER", C.DEFAULT_STT_PROVIDER).strip().lower()
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
                model=os.environ.get("PIPECAT_STT_MODEL", C.DEFAULT_STT_MODEL),
                language=Language.ZH,
                no_speech_prob=float(os.environ.get("PIPECAT_STT_NO_SPEECH_PROB", "0.4")),
            ),
        )

    if provider == "qwen":
        return QwenSTTService(
            base_url=os.environ.get("PIPECAT_QWEN_STT_BASE_URL", C.DEFAULT_QWEN_STT_BASE_URL),
            model=os.environ.get("PIPECAT_QWEN_STT_MODEL", C.DEFAULT_QWEN_STT_MODEL),
            language="zh",
        )

    raise ValueError(f"不支持的 STT provider: {provider}")


def create_tts_service():
    """根据环境变量创建语音合成服务。

    Returns:
        配置完成的 TTS 服务实例。

    Raises:
        ValueError: 如果 TTS provider 不支持。
    """
    provider = os.environ.get("PIPECAT_TTS_PROVIDER", C.DEFAULT_TTS_PROVIDER).strip().lower()
    if provider == "mimo":
        return MiMoTTSService(
            api_key=os.environ.get("MIMO_API_KEY"),
            voice=os.environ.get("PIPECAT_TTS_VOICE", C.DEFAULT_MIMO_TTS_VOICE),
            language=os.environ.get("PIPECAT_TTS_LANGUAGE", C.DEFAULT_MIMO_TTS_LANGUAGE),
        )

    if provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService
        from pipecat.transcriptions.language import Language

        lang_value = os.environ.get("PIPECAT_TTS_LANGUAGE", C.DEFAULT_KOKORO_TTS_LANGUAGE)
        language = Language(lang_value.strip().lower().replace("-", "_"))
        return KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice=os.environ.get("PIPECAT_TTS_VOICE", C.DEFAULT_KOKORO_TTS_VOICE),
                language=language,
            ),
        )

    if provider == "piper":
        from pipecat.services.piper.tts import PiperTTSService

        return PiperTTSService(
            settings=PiperTTSService.Settings(
                voice=os.environ.get("PIPECAT_TTS_VOICE", C.DEFAULT_PIPER_TTS_VOICE),
            ),
        )

    if provider == "voxcpm":
        return VoxCPMTTSService(
            base_url=os.environ.get("PIPECAT_VOXCPM_URL", C.DEFAULT_VOXCPM_TTS_URL),
            model=C.DEFAULT_VOXCPM_TTS_MODEL,
            voice=C.DEFAULT_VOXCPM_TTS_VOICE,
            ref_audio=_load_audio_data_uri(r"voice_samples/voice-preview-1-bingtang.wav"),
            ref_text="你好呀，我是冰糖，刚刚路过一家小店，闻到面包的味道，突然觉得好幸福呀",
            seed=C.DEFAULT_VOXCPM_TTS_SEED,
        )

    raise ValueError(f"不支持的 TTS provider: {provider}")


def create_llm_service():
    """根据环境变量创建大模型服务。

    使用 OpenAI 兼容 API 接口，默认为 MiMo 模型。

    Returns:
        配置完成的 LLM 服务实例。

    Raises:
        ValueError: 缺少 API key 时抛出。
    """
    api_key = os.environ.get("PIPECAT_LLM_API_KEY") or os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise ValueError("需要设置 MIMO_API_KEY 或 PIPECAT_LLM_API_KEY 环境变量")

    base_url = os.environ.get("PIPECAT_LLM_BASE_URL", C.DEFAULT_LLM_BASE_URL)
    model = os.environ.get("PIPECAT_LLM_MODEL", C.DEFAULT_LLM_MODEL)

    return OpenAILLMService(
        api_key=api_key,
        base_url=base_url,
        settings=OpenAILLMService.Settings(
            model=model,
            system_instruction=C.SYSTEM_PROMPT,
        ),
    )


def get_llm_credentials() -> tuple[str, str, str]:
    """获取 LLM 的 (api_key, base_url, model)，供记忆模块复用。

    Returns:
        元组 ``(api_key, base_url, model)``。

    Raises:
        ValueError: 缺少 API key 时抛出。
    """
    api_key = os.environ.get("PIPECAT_LLM_API_KEY") or os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise ValueError("需要设置 MIMO_API_KEY 或 PIPECAT_LLM_API_KEY 环境变量")
    base_url = os.environ.get("PIPECAT_LLM_BASE_URL", C.DEFAULT_LLM_BASE_URL)
    model = os.environ.get("PIPECAT_LLM_MODEL", C.DEFAULT_LLM_MODEL)
    return api_key, base_url, model
