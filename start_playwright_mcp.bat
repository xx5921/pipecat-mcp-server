@echo off
REM Playwright MCP Server launcher
REM Provides 23 browser automation tools: web browsing, screenshots, etc.
REM bot_standalone.py auto-connects via PIPECAT_MCP_SERVERS env var
REM Uses msedge browser on port 9093, streamable-http mode. Add --headless for headless mode.
npx @playwright/mcp --port 9093 --browser msedge --host localhost
