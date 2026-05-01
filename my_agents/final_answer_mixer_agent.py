from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from agent_state import AgentState
from common.llm import my_llm, get_top_k_history
import json


class FinalAnswerMixerAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_question = agent_state['input']
        cypher_answer = agent_state.get('cypher_answer', '')
        llm_answer = agent_state.get('llm_direct_answer', '')
        messages = [
            SystemMessage(content=(
                "你是一个答案融合专家，负责将知识图谱的结构化答案与大模型的开放式回答进行加权融合。"
                "知识图谱权重0.7，大模型权重0.3。"
                "请生成一段完整、流畅、自然的回答。"
            )),
            HumanMessage(content=f"用户问题：{user_question}\n\n知识图谱答案：{cypher_answer}\n\n大模型答案：{llm_answer}")
        ]
        response = my_llm.invoke(messages).content
        agent_state['output'] = response
        return agent_state
