from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from agent_state import AgentState
from common.llm import my_llm, get_top_k_history


class TCMQuestionClassifierAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        session_id = agent_state.get("session_id", "default")
        past_messages = get_top_k_history(session_id)
        messages = [
            SystemMessage(content=(
                "你是一个中医问题分类助手，专门判断用户的问题是否与中医有关。"
                "中医问题包括但不限于：中药、方剂、疾病、功效、症状、归经、性味等。"
                "非中医问题包括：现代西医、情感、生活常识、法律、娱乐等。"
                "请你只回答：是 或 否。不要补充其他内容。"
            ))] + past_messages + [
            HumanMessage(content=agent_state['input'])
        ]
        response = my_llm.invoke(messages).content.strip()
        agent_state['is_tcm_question'] = "是" in response
        return agent_state
