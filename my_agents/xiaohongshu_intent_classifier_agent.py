from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from agent_state import AgentState
from common.llm import my_llm


class XiaohongshuIntentClassifierAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_input = agent_state['input']
        messages = [
            SystemMessage(content=(
                "你是一个小红书内容意图识别专家。"
                "判断用户是否想在小红书平台发布中医科普内容。"
                "意图关键词：发笔记、写文章、分享到小红书、帮我发、发布、分享给大家。"
                "只回答：是 或 否。"
            )),
            HumanMessage(content=user_input)
        ]
        response = my_llm.invoke(messages).content.strip()
        agent_state['is_has_xhs_intent'] = "是" in response
        return agent_state
