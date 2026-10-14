from typing import Callable, Any

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, before_agent, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT

from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts


@wrap_tool_call()
def monitor_tool(
        # 请求的数据
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:  # 工具执行的监控

    logger.info(f"[tool_monitor] 执行工具：{request.tool_call['name']}")
    logger.info(f"[tool_monitor] 传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool_monitor] 工具：{request.tool_call['name']} 调用成功")
        # 只要模型调用了 fill_context_for_report 工具，就将 report 改为 true 进行标记注入
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context['report'] = True
        # 返回工具执行的操作
        return result
    except Exception as e:
        logger.error(f"[tool_monitor] 工具：{request.tool_call['name']} 调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        # 整个 Agent 智能体的状态记录
        state: AgentState,
        # 记录agent执行过程中的上下文
        runtime: Runtime[ContextT]
):  # 在模型执行前输出日志
    logger.info(f"[log_before_model] 即将调用模型，带有{len(state['messages'])}条消息")

    logger.debug(f"[log_before_model] {type(state['messages'][-1].__ne__)} | {state['messages'][-1].content}")

    # 继续执行，不需要跳转
    return None

@dynamic_prompt     # 每次生成提示器之前，调用此函数
def report_prompt_switch(request: ModelRequest):  # 动态切换提示词
    is_report = request.runtime.context.get('report', False)

    # 是报告生成场景
    if is_report:
        return load_report_prompts()

    return load_system_prompts()


