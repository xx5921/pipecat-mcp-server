#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat 独立语音助手启动脚本（冰糖）。

通过 .env 配置 STT/TTS/LLM 服务，启动一个支持语音对话的 Pipecat 机器人。
级联流水线：语音识别 → 大模型 → 语音合成

实现拆分在 ``pipecat_mcp_server.standalone`` 子包，本文件只负责入口。
记忆管理拆分在 ``pipecat_mcp_server.memory`` 子包。

运行方式：

    uv run python bot_standalone.py

环境变量配置（.env）：

- MIMO_API_KEY: MiMo API 密钥
- PIPECAT_STT_PROVIDER: STT 服务商（mimo / whisper / qwen，默认 mimo）
- PIPECAT_STT_MODEL: Whisper 模型名（tiny / base / small / medium / large-v3，默认 medium）
- PIPECAT_STT_DEVICE: Whisper 推理设备（cuda / cpu / auto，默认 cuda）
- PIPECAT_STT_COMPUTE_TYPE: Whisper 计算精度（float16 / int8_float16 / default，默认 float16）
- PIPECAT_STT_NO_SPEECH_PROB: 非语音概率阈值（默认 0.4）
- PIPECAT_QWEN_STT_BASE_URL: Qwen3-ASR 服务地址（默认 http://100.84.59.58:8200/v1）
- PIPECAT_QWEN_STT_MODEL: Qwen3-ASR 模型名（默认 Qwen/Qwen3-ASR-1.7B）
- PIPECAT_TTS_PROVIDER: TTS 服务商（mimo / kokoro / piper / voxcpm，默认 mimo）
- PIPECAT_TTS_VOICE: TTS 音色
- PIPECAT_TTS_LANGUAGE: TTS 语言代码
- PIPECAT_VOXCPM_URL: VoxCPM 服务地址
- PIPECAT_LLM_BASE_URL: LLM API 地址（默认 https://token-plan-cn.xiaomimimo.com/v1）
- PIPECAT_LLM_MODEL: LLM 模型名（默认 mimo-v2.5）
- PIPECAT_LLM_API_KEY: LLM API 密钥（为空时回退到 MIMO_API_KEY）
- PIPECAT_WAKE_WORD: 唤醒词（逗号分隔，为空则始终唤醒）
- PIPECAT_WAKE_TIMEOUT: 唤醒超时时间（秒，默认 60）
- PIPECAT_MCP_SERVERS: MCP 服务地址（逗号分隔）

记忆模块（可选）：

- PIPECAT_MEMORY_ENABLED: 是否启用记忆（1/0，默认 1）
- PIPECAT_MEMORY_DIR: 记忆文件目录（默认 memory）
- PIPECAT_MEMORY_FILE: 记忆文件名（默认 bingtang.json）
- PIPECAT_MEMORY_RECENT_TURNS: 短期窗口保留消息条数（默认 20）
- PIPECAT_MEMORY_FACT_EVERY: 隔多少完整回合提取一次事实（默认 5）
- PIPECAT_MEMORY_COMPRESS_BATCH: 累积多少条溢出消息后触发一次摘要压缩（默认 20）
- PIPECAT_LLM_MEMORY_MODEL: 记忆专用模型（默认跟主对话同模型；可设更便宜的小模型）
"""

from pipecat.runner.run import main
from pipecat.runner.types import RunnerArguments

from pipecat_mcp_server.standalone.runtime import bot as _run_bot


async def bot(runner_args: RunnerArguments):
    """机器人主入口，由 Pipecat runner 通过 ``__main__`` 自动发现。

    Args:
        runner_args: Pipecat 运行器传来的配置参数。
    """
    await _run_bot(runner_args)


if __name__ == "__main__":
    main()
