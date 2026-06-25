#!/usr/bin/env python3
"""Qwen3-ASR 语音识别脚本，调用本地 Docker 服务的 OpenAI 兼容接口。

直接修改下方配置区的参数后运行: python recognize.py
"""

import base64
import json
import sys
from pathlib import Path

import requests

# ============================================================
# 配置区 —— 修改以下参数
# ============================================================

AUDIO = "voice_samples/voice-preview-1-bingtang.wav"

# 强制指定语言，None 表示自动检测
# 可选值: None(自动), "Chinese", "English"
LANGUAGE = None

# 请求超时时间（秒）
TIMEOUT = 120

# 结果输出格式: True=JSON, False=文本
OUTPUT_JSON = False

# API 地址（一般无需修改）
API_BASE = "http://100.84.59.58:8200/v1"

# ============================================================
# 以下为脚本逻辑，一般无需修改
# ============================================================


def _read_audio_as_data_url(file_path: str) -> str:
    """读取本地音频文件并编码为 data URL。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    suffix = path.suffix.lower().lstrip(".")
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
                 "ogg": "audio/ogg", "m4a": "audio/mp4", "aac": "audio/aac"}
    mime = mime_map.get(suffix, "audio/wav")

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _build_audio_content(audio: str) -> dict:
    """根据输入类型构建 audio content 对象。"""
    if audio.startswith(("http://", "https://", "data:")):
        return {"type": "audio_url", "audio_url": {"url": audio}}
    return {"type": "audio_url", "audio_url": {"url": _read_audio_as_data_url(audio)}}


def _parse_asr_output(content: str) -> dict:
    """解析 ASR 返回内容，提取语言和文本。"""
    if not content:
        return {"language": "unknown", "text": ""}

    lang = "unknown"
    text = content

    if "<asr_text>" in content:
        prefix, text = content.split("<asr_text>", 1)
        lang_part = prefix.replace("language", "").strip()
        lang = lang_part if lang_part else "unknown"
    elif content.startswith("language "):
        parts = content.split("\n", 1)
        lang = parts[0].replace("language ", "").strip()
        text = parts[1] if len(parts) > 1 else ""

    return {"language": lang, "text": text.strip()}


def recognize(audio: str, language: str = None, timeout: int = 120) -> dict:
    """调用 Qwen3-ASR 接口识别语音。

    Args:
        audio: 本地音频文件路径或音频 URL。
        language: 强制指定语言，None 表示自动检测。
        timeout: 请求超时时间（秒）。

    Returns:
        包含 language, text, usage 的字典。
    """
    content_parts = [_build_audio_content(audio)]

    if language:
        content_parts.append({"type": "text", "text": f"Transcribe to {language}:"})

    payload = {
        "messages": [{"role": "user", "content": content_parts}],
    }

    resp = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    raw_content = choice["message"]["content"]
    parsed = _parse_asr_output(raw_content)

    return {
        "language": parsed["language"],
        "text": parsed["text"],
        "usage": data.get("usage", {}),
    }


def main():
    try:
        result = recognize(AUDIO, language=LANGUAGE, timeout=TIMEOUT)
    except requests.ConnectionError:
        print("错误: 无法连接到 Qwen3-ASR 服务 (http://localhost:8200)", file=sys.stderr)
        sys.exit(1)
    except requests.Timeout:
        print("错误: 请求超时", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    if OUTPUT_JSON:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"语言: {result['language']}")
        print(f"文本: {result['text']}")


if __name__ == "__main__":
    main()
