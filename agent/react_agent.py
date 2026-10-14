from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize, get_weather, get_user_id, get_user_location,
    get_abs_path, get_current_month, fill_context_for_report, fetch_external_data
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize, fill_context_for_report, fetch_external_data,
                get_weather, get_user_id, get_user_location, get_abs_path, get_current_month
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                HumanMessage(content=f"{query}")
            ]
        }

        # 第三个参数 context 就是上下文 runtime 中的信息，就是我们提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk['messages'][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    # for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
    #     print(chunk, end='', flush=True)

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end='', flush=True)
