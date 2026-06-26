"""独立语音助手（bot_standalone）的支撑子包。

把原本散落在 ``bot_standalone.py`` 里的常量、唤醒词逻辑、MCP 加载、
服务工厂、内置工具、运行时主流程拆成独立模块，便于维护和复用。
本子包完全独立于 MCP 服务路径（``server.py`` / ``agent.py`` / ``bot.py``），
新增它不影响 pip 安装后 ``pipecat-mcp-server`` 命令的行为。
"""

from .runtime import bot, run_bot

__all__ = ["run_bot", "bot"]
