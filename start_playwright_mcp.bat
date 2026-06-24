@echo off
REM Playwright MCP Server 启动脚本
REM 浏览器自动化 MCP 服务，提供网页浏览、截图等工具

REM 使用 Chromium 浏览器，端口 9091，SSE 模式
npx @playwright/mcp --port 9091 --browser chromium --host localhost
