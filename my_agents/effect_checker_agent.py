from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm
import json


class EffectOutput(BaseModel):
    is_effect_question: bool = Field(description="是否涉及功效相关问题")
    entities: List[str] = Field(default_factory=list, description="提取出的功效实体列表")


class EffectCheckerAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_question = agent_state['input']
        prompt = f"""你是一个中医功效识别专家，从以下用户问题中提取所有功效相关实体。
只返回 JSON 格式：{{"is_effect_question": true/false, "entities": ["功效1", "功效2"]}}
问题：{user_question}"""
        response = my_llm.invoke([
            SystemMessage(content="你是一个中医功效识别专家，擅长从文本中提取功效实体。"),
            HumanMessage(content=prompt)
        ])
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:-1])
            parsed = json.loads(content)
            agent_state['is_effect_question'] = parsed.get('is_effect_question', False)
            agent_state['effect_entities'] = parsed.get('entities', [])
        except Exception:
            agent_state['is_effect_question'] = False
            agent_state['effect_entities'] = []
        return agent_state
