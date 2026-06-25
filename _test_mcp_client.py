"""测试 MCP 客户端 - 使用 trust_env=False 绕过代理。"""
import asyncio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://localhost:9092/mcp"


async def main():
    http_client = httpx.AsyncClient(trust_env=False, timeout=30.0)
    print(f"连接 MCP 服务: {MCP_URL}")
    try:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"\n已发现 {len(tools.tools)} 个工具:\n")
                for tool in tools.tools:
                    schema = tool.inputSchema or {}
                    props = list(schema.get("properties", {}).keys())
                    print(f"  - {tool.name}: {props}")

                print(f"\n总计 {len(tools.tools)} 个工具")

                print("\n--- 测试 browser_navigate ---")
                result = await session.call_tool("browser_navigate", {"url": "https://www.baidu.com"})
                for block in result.content:
                    if hasattr(block, "text"):
                        print(block.text[:300])
                print("--- OK ---")
    finally:
        await http_client.aclose()


asyncio.run(main())
