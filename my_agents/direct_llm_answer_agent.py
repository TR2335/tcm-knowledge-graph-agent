from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm


class DirectLLMAnswerAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_question = agent_state['input']
        cypher_result = agent_state.get('cypher_result', [])
        if cypher_result:
            answer = agent_state.get('cypher_answer', '')
            if answer:
                agent_state['llm_direct_answer'] = answer
                return agent_state
        messages = [
            SystemMessage(content="你是一个中医知识专家，请根据你的知识回答用户问题。"),
            HumanMessage(content=user_question)
        ]
        response = my_llm.invoke(messages).content
        agent_state['llm_direct_answer'] = response
        if not agent_state.get('output'):
            agent_state['output'] = response
        return agent_state
