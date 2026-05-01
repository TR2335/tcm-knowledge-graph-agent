from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm, get_top_k_history
# result → 瑞扎特
# parameters → 坡瑞米特兹

# 增强query能力 拿到历史对话记录 作为上下文 给模型 生成符合要求的查询语句
# 直接用大模型回答用户问题，不依赖图谱 7
class DirectLLMAnswerAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        """直接用大模型回答用户问题，不依赖图谱"""
        session_id = agent_state.get("session_id", "default")
        # 拿到最近 k 条历史
        past_messages = get_top_k_history(session_id)

        messages = [SystemMessage(content=(
            "你是一个博学而通俗的中医助手，擅长用简洁易懂的语言回答各种中医相关问题。"
            "用户提问的问题，可能在中医领域，也可能不在中医领域。"
            "如果用户的问题超出中医领域，也请尽可能给出有参考价值的自然语言解释。"
            "回答要自然、准确、简明扼要。"
        ))] + past_messages + [HumanMessage(content=agent_state['input'])]

        response = my_llm.invoke(messages).content.strip()
        agent_state['llm_direct_answer'] = response
        agent_state['output'] = response
        if not agent_state.get('cypher_answer', None):
            agent_state['cypher_answer'] = ''
        print(agent_state)
        return agent_state
