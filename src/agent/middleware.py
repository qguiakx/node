from typing import Any, Callable, Awaitable
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command


class LoggingMiddleware(AgentMiddleware):
    """
    日志记录中间件 - 同时支持同步和异步调用
    解决 FastAPI 中使用 astream()/ainvoke() 时的 NotImplementedError
    """

    # ==================== 节点式钩子（同步） ====================

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[before_model] 模型即将调用，并附带 {len(state['messages'])} 条消息")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[after_model] 模型调用完成，当前共 {len(state['messages'])} 条消息")
        return None

    # ==================== 节点式钩子（异步） ====================

    async def abefore_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print(f"[abefore_model] 模型即将调用，并附带 {len(state['messages'])} 条消息")
        return None

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print(f"[aafter_model] 模型调用完成，当前共 {len(state['messages'])} 条消息")
        return None

    # ==================== 包装式钩子：模型调用（同步） ====================

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        print("[wrap_model_call] 模型调用开始")
        try:
            response = handler(request)
            print("[wrap_model_call] 模型调用成功")
            return response
        except Exception as e:
            print(f"[wrap_model_call] 模型调用失败: {e}")
            raise

    # ==================== 包装式钩子：模型调用（异步） ====================
    # ⚠️ 必须实现！否则在 FastAPI/astream 中会报 NotImplementedError

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        print("[awrap_model_call] 模型调用开始")
        try:
            response = await handler(request)   # ← 关键：必须 await
            print("[awrap_model_call] 模型调用成功")
            return response
        except Exception as e:
            print(f"[awrap_model_call] 模型调用失败: {e}")
            raise

    # ==================== 包装式钩子：工具调用（同步） ====================

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = self._get_tool_name(request)
        tool_args = self._get_tool_args(request)

        print(f"[wrap_tool_call] 工具名称: {tool_name}")
        print(f"[wrap_tool_call] 工具参数: {tool_args}")

        try:
            result = handler(request)
            print("[wrap_tool_call] 工具执行成功")
            return result
        except Exception as e:
            print(f"[wrap_tool_call] 工具执行失败: {e}")
            raise

    # ==================== 包装式钩子：工具调用（异步） ====================

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = self._get_tool_name(request)
        tool_args = self._get_tool_args(request)

        print(f"[awrap_tool_call] 工具名称: {tool_name}")
        print(f"[awrap_tool_call] 工具参数: {tool_args}")

        try:
            result = await handler(request)   # ← 关键：必须 await
            print("[awrap_tool_call] 工具执行成功")
            return result
        except Exception as e:
            print(f"[awrap_tool_call] 工具执行失败: {e}")
            raise

