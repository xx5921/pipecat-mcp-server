"""内置（非 MCP）工具函数。

每个工具是一个签名兼容 ``FunctionCallParams`` 的 async 函数，
通过 ``BOT_TOOLS`` 列表统一注册到 LLMContext。
"""

from datetime import datetime, timedelta, timezone

from pipecat.services.llm_service import FunctionCallParams

# 北京时区
_TZ_CN = timezone(timedelta(hours=8))


async def tool_get_current_time(params: FunctionCallParams):
    """获取当前北京时间。"""
    now = datetime.now(_TZ_CN)
    await params.result_callback({
        "datetime": now.strftime("%Y年%m月%d日 %H:%M:%S"),
        "weekday": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()],
    })


# 工具列表
BOT_TOOLS = [tool_get_current_time]
