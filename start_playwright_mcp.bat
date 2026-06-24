@echo off
REM Playwright MCP Server 启动脚本
REM 浏览器自动化 MCP 服务，提供网页浏览、截图等 23 个工具
REM 启动后 bot_standalone.py 会自动通过 PIPECAT_MCP_SERVERS 连接

REM 使用 Chromium 浏览器，端口 9092，streamable-http 模式
npx @playwright/mcp --port 9093 --browser msedge --headless --host localhost
