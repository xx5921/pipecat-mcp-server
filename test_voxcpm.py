import requests

# 1. 配置接口地址和参数
# 注意：根据你的 Docker 启动映射，如果外部宿主机端口是 8000，请保持 8000；如果是 8100 请修改此处。
URL = "http://100.84.59.58:8100/v1/audio/speech"

headers = {
    "Content-Type": "application/json"
}

payload = { # 你的模型名称
    "input": "你好啊，今天是开心的一天",  # 想要转换的文本
    "voice": "default",  # 声音音色
    "response_format": "wav"  # 支持 wav, mp3 等格式
}

print(generosity_msg := f"正在发送请求到 {URL}，请稍候...")

try:
    # 2. 发送 POST 请求
    response = requests.post(URL, json=payload, headers=headers)

    # 3. 检查并保存音频文件
    if response.status_code == 200:
        output_filename = "output.wav"
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"🎉 语音生成成功！音频文件已保存至当前目录下的: {output_filename}")
    else:
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print(f"错误详情: {response.text}")

except Exception as e:
    print(f"💥 连接服务器出错: {e}")