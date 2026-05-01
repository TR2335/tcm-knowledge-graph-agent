# File: MyAgents/FinalAnswerMixerAgent.py
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage

from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm, get_top_k_history

# 8
class FinalAnswerMixerAgent(Runnable):
    """融合图谱回答与LLM回答，利用LLM进行补充增强输出"""



    '''
    将两个来源的答案合并成一个最终回答：
  - 回答 A：cypher_answer（Neo4j 图谱查询结果，更权威、结构化）
  - 回答 B：llm_direct_answer（LLM 直接回答，更通俗、丰富）
    '''
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_question = agent_state.get("input", "")
        cypher_answer = agent_state.get("cypher_answer", "")
        llm_direct_answer = agent_state.get("llm_direct_answer", "")

        system_prompt = """
        你是一个中医问答专家，需要将两个来源的回答进行融合：

        - 回答 A 来自中医知识图谱，通常更权威、准确、结构化；
        - 回答 B 来自大语言模型，可能更通俗、丰富。

        请遵循以下融合策略：

        1. **始终以 A 为主要答案来源，保留其原始结构与表述。**
        2. **仅当 B 内容能对 A 做出“明确补充”（如现代解释、注意事项、通俗举例）时，可将补充信息适当加入，不得覆盖或替换 A。**
        3. **若 A 完全为空或内容极差（如仅回答“无法回答”），则退而使用 B 答案。**
        4. 最终答案应清晰自然、专业准确，突出图谱权威性，适当增强可读性。

        请直接输出整合后的最终回答，不要说明答案来源。
        """.strip()

        content = f"""用户提问：{user_question}

        回答 A（图谱结果）：
        {cypher_answer}
        
        回答 B（模型结果）：
        {llm_direct_answer}
        """
        # session_id 默认是 default
        session_id = agent_state.get("session_id", "default")
        # 拿到最近 k 条历史
        past_messages = get_top_k_history(session_id)
        # 构建消息列表 system_prompt 是系统提示 + past_messages 是历史消息 + content 是当前问题
        messages = ([
                        SystemMessage(content=system_prompt)] + past_messages + [
                        HumanMessage(content=content.strip())
                    ])

        print("🤖 正在让 LLM 整合两个答案...")
        final_answer = my_llm.invoke(messages).content.strip()

        agent_state["output"] = final_answer
        print("✅ LLM 整合输出：", final_answer)
        return agent_state
