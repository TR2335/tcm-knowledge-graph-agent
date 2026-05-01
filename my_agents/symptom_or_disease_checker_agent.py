from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
from agent_state import AgentState
from common.llm import my_llm


class SymptomDiseaseOutput(BaseModel):
    is_symptom_or_disease: bool = Field(description="是否包含症状或疾病实体")
    entities: List[str] = Field(default_factory=list, description="提取出的症状或疾病实体列表")


class SymptomOrDiseaseCheckerAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_question = agent_state['input']
        prompt = f"""你是一个中医实体识别专家，从以下用户问题中提取所有症状或疾病实体。
只返回 JSON 格式：{{"is_symptom_or_disease": true/false, "entities": ["实体1", "实体2"]}}
问题：{user_question}"""
        response = my_llm.invoke([
            SystemMessage(content="你是一个中医实体识别专家，擅长从文本中提取症状或疾病。"),
            HumanMessage(content=prompt)
        ])
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:-1])
            parsed = json.loads(content)
            agent_state['is_symptom_or_disease'] = parsed.get('is_symptom_or_disease', False)
            agent_state['symptom_or_disease_entities'] = parsed.get('entities', [])
        except Exception:
            agent_state['is_symptom_or_disease'] = False
            agent_state['symptom_or_disease_entities'] = []
        return agent_state
