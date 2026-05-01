from typing import List
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm, get_top_k_history

#todo is_symptom_or_disease: bool：是 True/False，代表用户问题是否涉及症状 / 疾病
# entities: List[str]：列表，存放提取出来的症状或疾病名称
# 作用：强制 AI 按这个结构输出，程序才能解析
class SymptomOrDiseaseCheckOutput(BaseModel):
    is_symptom_or_disease: bool
    entities: List[str]

# 实体获取 检查症状或疾病 2
class SymptomOrDiseaseCheckerAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        """判断一个问题是否涉及症状或疾病，并提取相关实体"""
        # 用class的格式化指令解析器 json 格式化
        parser = PydanticOutputParser(pydantic_object=SymptomOrDiseaseCheckOutput)
        # 生成格式化指令
        format_instructions = parser.get_format_instructions()
        # get_format_instructions 生成格式化指令
        session_id = agent_state.get("session_id", "default")
        # 从状态中获取会话 ID，默认值为 "default"
        # 从状态里拿会话 ID，没有就用 default
        # 作用：区分不同用户的对话

        # 拿到最近 k=5 条历史
        past_messages = get_top_k_history(session_id)
        messages = [
                       SystemMessage(content=(
                           "你是一个中医问句分析助手，任务有两部分：\n"
                           "1. 判断用户的问题是否涉及“症状”或“疾病”。只要包含任一症状（如咳嗽、头痛、失眠）或疾病（如感冒、糖尿病、高血压），就算相关；否则为不相关。\n"
                           "2. 如果相关，请抽取其中涉及的症状或疾病名称，并以列表形式输出。\n\n"
                           "请你严格按照以下格式返回结果：\n"
                           f"{format_instructions}"
                       ))] + past_messages + [
                       HumanMessage(content=agent_state['input'])
                   ]
        '''
        SystemMessage：给 AI 设定角色
        中医问句分析助手
        任务 1：判断是否有症状 / 疾病
        任务 2：提取症状 / 疾病名词
        强制按格式返回
        past_messages：加上历史对话
        HumanMessage：加上用户当前问题
        '''
        raw_output = my_llm.invoke(messages).content.strip()
        # 调用 LLM 生成原始输出
        parsed_output = parser.parse(raw_output)
        # 解析原始输出，生成结构化输出
        agent_state['is_symptom_or_disease'] = parsed_output.is_symptom_or_disease
        #把解析后的症状或疾病名称列表，赋值给状态
        agent_state['symptom_or_disease_entities'] = parsed_output.entities
        # 打印状态，方便调试
        print(agent_state)
        return agent_state
