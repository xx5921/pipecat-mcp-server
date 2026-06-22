import base64
import os

import requests
import pyaudio

# 1. 配置参数（完全对齐官方 48 kHz s16le 标准）
MODEL_NAME = "openbmb/VoxCPM2"
URL = "http://100.84.59.58:8100/v1/audio/speech"

FORMAT = pyaudio.paInt16  # s16le (16位有符号整数)
CHANNELS = 1  # -c 1 (单声道)
RATE = 48000  # -r 48000 (48 kHz 采样率)

def get_audio_b64(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"没有找到参考音频文件: {path}，请先放一个短音频在同目录下。")
    with open(path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/wav;base64,{b64_data}"

payload = {
    "model": MODEL_NAME,
    "input": "原生风格的背景色不会跟随 QSS 主题，导致出现白色背景与主题不协调、遮挡文本的情况。",
    "voice": "default",
    # "ref_audio": get_audio_b64("output.wav"),  # 👈 真正决定音色的核心参数,
    # "ref_text": "你好啊，今天是开心的一天",
    "seed": 2028,
    "stream": True,  # 开启流式
    "response_format": "pcm"  # 👈 核心：改用官方要求的 pcm
}

headers = {"Content-Type": "application/json"}

# 2. 初始化本地音频播放器
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True)  # 设置为输出模式（播放）

print(f"🚀 正在连接流式音频服务器并请求 48kHz PCM...")

try:
    # 3. 发送请求，开启 requests 的流式接收
    with requests.post(URL, json=payload, headers=headers, stream=True) as response:
        if response.status_code == 200:
            print("🔊 正在实时解码并播放音频流 (TTS)...")

            # 每次读取 1024 字节的裸 PCM 数据，直接喂给声卡
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    stream.write(chunk)  # 👈 这一步相当于 Linux 的 | play -t raw ...
                    print(".", end="", flush=True)

            print("\n🎉 播放完毕！")
        else:
            print(f"\n❌ 服务器拒绝请求，状态码: {response.status_code}")
            print(f"详情: {response.text}")

except Exception as e:
    print(f"\n💥 发生错误: {e}")

finally:
    # 4. 释放声卡资源
    stream.stop_stream()
    stream.close()
    p.terminate()