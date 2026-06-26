"""MCP 工具加载与清理。

通过 MCP Streamable HTTP 协议连接一组 MCP 服务器，把每个服务器提供的工具
包装成 Pipecat 的 FunctionSchema，统一注册到 LLMContext。
bot 停止时通过 :func:`cleanup_mcp` 关闭所有会话。
"""

import httpx
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

# 内置工具名（避免和 MCP 工具重名）
_BUILTIN_TOOL_NAMES = {"get_current_time"}

# 全局 MCP 会话/连接管理器（bot 停止时统一清理）
_mcp_sessions: list[ClientSession] = []
_mcp_transports: list = []
_mcp_http_clients: list[httpx.AsyncClient] = []


async def load_mcp_tools(server_urls: list[str]) -> list[FunctionSchema]:
    """连接 MCP 服务器并动态注册其提供的工具。

    Args:
        server_urls: MCP 服务器端点列表，例如 ``["http://localhost:9091/mcp"]``。

    Returns:
        Pipecat FunctionSchema 工具列表（可能为空）。
    """
    tools: list[FunctionSchema] = []
    http_client = httpx.AsyncClient(trust_env=False, timeout=30.0)
    _mcp_http_clients.append(http_client)

    for url in server_urls:
        url = url.strip()
        if not url:
            continue
        logger.info(f"[MCP] 正在连接 MCP 服务: {url}")
        try:
            transport_ctx = streamable_http_client(url, http_client=http_client)
            read, write, _ = await transport_ctx.__aenter__()
            _mcp_transports.append(transport_ctx)

            session = ClientSession(read, write)
            await session.__aenter__()
            _mcp_sessions.append(session)

            await session.initialize()
            result = await session.list_tools()

            for tool in result.tools:
                # 避免与内置工具重名
                schema_name = (
                    f"mcp_{tool.name}" if tool.name in _BUILTIN_TOOL_NAMES else tool.name
                )
                input_schema = tool.inputSchema or {}
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])
                description = tool.description or ""

                handler = _make_handler(tool.name, session)
                schema = FunctionSchema(
                    name=schema_name,
                    description=description,
                    properties=properties,
                    required=required,
                    handler=handler,
                )
                tools.append(schema)
                logger.info(f"[MCP] 已注册工具: {schema_name} (来自 {url})")

            logger.info(f"[MCP] {url} 加载完成，共 {len(result.tools)} 个工具")
        except Exception as e:
            logger.warning(f"[MCP] 连接失败 {url}: {e}")

    return tools


def _make_handler(tool_name: str, session: ClientSession):
    """为单个 MCP 工具生成 Pipecat 兼容的 handler。

    Args:
        tool_name: MCP 工具原始名字。
        session: 已初始化的 MCP ClientSession。

    Returns:
        async handler，签名兼容 FunctionCallParams。
    """

    async def handler(params: FunctionCallParams):
        logger.info(f"[MCP] 调用工具: {tool_name}({dict(params.arguments)})")
        try:
            call_result = await session.call_tool(tool_name, dict(params.arguments))
            texts = []
            for block in call_result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            result_text = "\n".join(texts) if texts else str(call_result.content)
            await params.result_callback({"result": result_text})
        except Exception as e:
            logger.warning(f"[MCP] 工具调用失败 {tool_name}: {e}")
            await params.result_callback({"error": str(e)})

    return handler


async def cleanup_mcp() -> None:
    """关闭所有 MCP 客户端会话/连接/http client。"""
    for session in _mcp_sessions:
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            pass
    _mcp_sessions.clear()
    for transport in _mcp_transports:
        try:
            await transport.__aexit__(None, None, None)
        except Exception:
            pass
    _mcp_transports.clear()
    for client in _mcp_http_clients:
        try:
            await client.aclose()
        except Exception:
            pass
    _mcp_http_clients.clear()
