from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm


class XiaohongshuTCMPostAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_input = agent_state['input']
        messages = [
            SystemMessage(content=(
                "你是一个小红书中医科普内容创作者，根据用户输入生成吸引人的小红书笔记。"
                "包含：标题（emoji点缀）、正文（分点阐述，有小标题）、话题标签（3-5个）。"
                "风格：亲切、专业、有温度。字数：300-500字。"
            )),
            HumanMessage(content=user_input)
        ]
        response = my_llm.invoke(messages).content
        parts = response.split("\n", 1)
        if len(parts) > 1:
            agent_state['xiaohongshu_tcm_post_title'] = parts[0].replace("标题：", "").strip()
            agent_state['xiaohongshu_tcm_post_content'] = parts[1].strip()
        else:
            agent_state['xiaohongshu_tcm_post_title'] = "中医养生分享"
            agent_state['xiaohongshu_tcm_post_content'] = response
        agent_state['xiaohongshu_tcm_post_strategies'] = []
        return agent_state
