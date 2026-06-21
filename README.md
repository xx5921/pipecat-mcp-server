<h1><div align="center">
 <img alt="Pipecat MCP Server" width="300px" height="auto" src="https://github.com/pipecat-ai/pipecat-mcp-server/raw/refs/heads/main/pipecat.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-mcp-server)](https://pypi.org/project/pipecat-ai-mcp-server) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

# Pipecat MCP Server

Pipecat MCP Server 为你的 AI 助手（Claude Code / Codex CLI）赋予**语音交互能力**，基于 [Pipecat](https://github.com/pipecat-ai/pipecat) 实现。它兼容所有 [MCP](https://modelcontextprotocol.io/) 客户端。

**核心概念**：MCP Server 暴露语音和屏幕捕获工具给 AI 客户端，但它本身不提供麦克风和扬声器。音频输入输出由**独立的传输层**处理，默认使用 WebRTC，你可以通过浏览器连接到本地服务。

> AI 客户端（Claude Code、Codex）负责**控制对话**，不是音频设备。要听到、说出或看到，你需要通过音频传输层连接。

## 架构流程

```
你(浏览器) ──WebRTC──▶ Pipecat Agent (STT/TTS) ◀──MCP──▶ Claude Code / Codex CLI
   ▲                        ▲                                ▲
   │                        │                                │
   音频                  语音工具                       AI 大脑
 (听/说)           (listen/speak/start/stop)          (理解/决策)
```

## 环境要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 包管理器
- MiMo API Key（STT + TTS 云端服务，中文识别效果好于本地 Whisper）

## 安装

### 方式一：从 PyPI 安装

```bash
uv tool install pipecat-ai-mcp-server
```

### 方式二：克隆仓库本地安装

```bash
git clone https://github.com/xx5921/pipecat-mcp-server.git
cd pipecat-mcp-server
uv tool install -e .
```

### 配置环境变量

在项目目录创建 `.env` 文件：

```bash
# MiMo API Key（必填，用于语音识别和合成）
MIMO_API_KEY=你的MiMo_API_Key

# WebRTC Runner 配置（可选，以下为默认值）
PIPECAT_RUNNER_HOST=localhost
PIPECAT_RUNNER_PORT=7860
PIPECAT_RUNNER_TRANSPORT=webrtc
```

## 启动服务

```bash
pipecat-mcp-server
```

服务启动后：
- MCP Server 运行在 `http://localhost:9090/mcp`
- Pipecat Runner（WebRTC）运行在 `http://localhost:7860`

---

## 连接 Claude Code

### 步骤 1：添加 MCP Server

```bash
claude mcp add pipecat --transport http http://localhost:9090/mcp --scope user
```

Scope 选项：
- `local`：仅当前项目生效
- `user`：所有项目生效
- `project`：存储在项目的 `.mcp.json` 中

### 步骤 2：配置权限自动批准

创建 `.claude/settings.local.json`：

```json
{
  "permissions": {
    "allow": [
      "mcp__pipecat__start",
      "mcp__pipecat__listen",
      "mcp__pipecat__speak",
      "mcp__pipecat__stop",
      "mcp__pipecat__list_windows",
      "mcp__pipecat__screen_capture",
      "mcp__pipecat__capture_screenshot"
    ]
  }
}
```

### 步骤 3：启动语音对话

1. 确保 `pipecat-mcp-server` 已启动
2. 在浏览器打开 `http://localhost:7860`，点击连接（这是你的麦克风和扬声器）
3. 在 Claude Code 中说：**"开始语音对话"** 或直接说你想做的事

Claude 会自动调用以下流程：
1. `start()` → 启动 Pipecat 语音代理
2. `listen()` → 等待你说话，返回转录文字
3. Claude 思考并生成回复
4. `speak(text)` → TTS 播报回复
5. 循环 listen/speak 直到你说结束
6. `stop()` → 关闭语音通道

---

## 连接 Codex CLI

### 步骤 1：添加 MCP Server

```bash
codex mcp add pipecat --url http://localhost:9090/mcp
```

### 步骤 2：配置信任级别

在 Codex 中进入你的项目目录，Codex 会询问是否信任该目录。选择 `Yes`，这会在 `~/.codex/config.toml` 中添加：

```toml
[projects."/path/to/your/project"]
trust_level = "trusted"
```

### 步骤 3：启动语音对话

1. 确保 `pipecat-mcp-server` 已启动
2. 在浏览器打开 `http://localhost:7860`，点击连接
3. 在 Codex 中输入 `/talk` 或说 "开始语音对话"

---

## 屏幕捕获与分析

你可以把屏幕（或某个窗口）共享给 AI 助手，让它帮你分析看到的内容。

**可用工具：**
- `list_windows()` — 列出所有可捕获的窗口
- `screen_capture(window_id)` — 开始捕获指定窗口（不传则捕获全屏）
- `capture_screenshot()` — 截取当前画面并保存为图片

**使用示例：**
- "列出我打开的窗口" → 返回窗口列表
- "捕获我的浏览器窗口" → 开始流式传输该窗口
- "这个报错是什么原因？" → AI 分析你的屏幕画面
- "这个 UI 设计怎么样？" → AI 给你反馈

**支持平台：**
- **macOS** — ScreenCaptureKit，支持窗口级捕获
- **Linux (X11)** — Xlib 窗口和全屏捕获
- **Windows** — 全屏捕获

---

## 自定义服务

### 切换 STT / TTS

在 `.env` 中通过环境变量切换语音识别和语音合成服务：

```bash
# STT provider: mimo / whisper
PIPECAT_STT_PROVIDER=whisper
# Whisper model: tiny / base / small / medium / large-v3
PIPECAT_STT_MODEL=medium
PIPECAT_STT_NO_SPEECH_PROB=0.4

# TTS provider: mimo / kokoro / piper
PIPECAT_TTS_PROVIDER=piper
# MiMo example: mimo_default / 冰糖 / 茉莉 / 苏打 / 白桦 / Mia / Chloe / Milo / Dean
# Kokoro example: af_heart
# Piper example: zh_CN-huayan-medium
PIPECAT_TTS_VOICE=zh_CN-huayan-medium
# Kokoro af_heart uses en; MiMo/Piper Chinese voices usually use zh.
PIPECAT_TTS_LANGUAGE=zh
```

- `mimo`：小米云端服务，中文识别和合成效果较好，需要 `MIMO_API_KEY`。
- `whisper`：本地 Whisper 语音识别，免费，首次启动会自动下载模型。
- `kokoro`：本地 Kokoro ONNX 语音合成，免费，首次启动会自动下载模型。
- `piper`：本地 Piper 语音合成，免费，首次启动会自动下载指定音色模型。

如果切换到 `PIPECAT_TTS_PROVIDER=kokoro` 且使用 `af_heart`，请把 `PIPECAT_TTS_LANGUAGE` 改成 `en`，否则 Kokoro 的 espeak 后端会报 `zh` 不支持。

### 切换传输层

默认 WebRTC。如需使用 Daily 房间，在 `.env` 中设置：

```bash
PIPECAT_RUNNER_TRANSPORT=daily
DAILY_API_KEY=你的Daily_API_Key
DAILY_ROOM_URL=你的Daily房间地址
```

---

## 常见问题

**Q: 说话后听到两次回复？**
A: Pipeline 中不要放置 LLM 服务。本项目的架构中，AI 客户端（Claude/Codex）是"大脑"，Pipeline 只需要 STT + TTS。

**Q: 浏览器界面上看不到 AI 的文字回复？**
A: TTS 会消费 `LLMTextFrame` 并输出音频帧，文字无法到达 UI。`agent.py` 已修复此问题：`speak()` 会同时推送文字到 `assistant_aggregator` 用于 UI 显示。

**Q: 如何修改 TTS 音色？**
A: 可选音色：`mimo_default`、`冰糖`、`茉莉`、`苏打`、`白桦`、`Mia`、`Chloe`、`Milo`、`Dean`。在 `agent.py` 的 `_create_tts_service()` 中修改 `voice` 参数即可。

---

## 更多资源

- [Pipecat 文档](https://docs.pipecat.ai/)
- [Pipecat Discord](https://discord.gg/pipecat)
- [MCP 协议](https://modelcontextprotocol.io/)
- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code)
