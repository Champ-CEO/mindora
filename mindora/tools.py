from typing import Any, List

from langchain.tools import tool
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from rich.pretty import pprint

from mindora.memory import Memory, create_memory_manager


def get_available_tools() -> List[BaseTool]:
    return [save_memory]


def call_tool(tool_call: ToolCall) -> Any:
    tools_by_name = {tool.name: tool for tool in get_available_tools()}
    tool = tools_by_name[tool_call["name"]]
    response = tool.invoke(tool_call["args"])

    print("Tool call:")
    pprint(tool_call)
    return response


@tool
def save_memory(content: str, importance: int, config: RunnableConfig) -> str:
    """Save important information about the user.

    Args:
        content: The memory content to save
        importance: Importance rating from 1 (low) to 10 (high)
        config: Runtime configuration

    Returns:
        Confirmation message
    """

    user_id = config["configurable"].get("user_id")

    memory = Memory(content=content, user_id=user_id, importance=importance)
    create_memory_manager().save_memory(memory)
    return f"Memory saved: {content}"
