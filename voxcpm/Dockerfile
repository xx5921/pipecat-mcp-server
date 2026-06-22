FROM vllm/vllm-openai:latest

# 安装系统依赖 git，并把所有可能用到的 Python 依赖及 vllm-omni 一并安装
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir voxcpm soundfile httpx ninja && \
    pip install --no-cache-dir git+htt