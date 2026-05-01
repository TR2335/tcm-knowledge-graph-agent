# File: MyAgents/EffectCheckerAgent.py
from typing import List
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm, get_top_k_history

# 解析 LLM 输出: 4
# 被 EffectCheckerAgent 用于解析 LLM 的输出结果 把输出结构化
# 构建包含系统提示和用户问题的消息
# is_effect_question 是否涉及功效
# entities 功效实体列表
class EffectCheckOutput(BaseModel):
    is_effect_question: bool
    entities: List[str]

'''
为什么 分开做的 疾病词 与 功效识别
疾病/症状：问题导向（What's wrong）
功效：解决方案导向（What it can do）

不同的语义空间需要不同的向量索引
疾病/症状的语义相似度计算与功效不同
分离索引提高检索精度和效率 

更便于维护分析使用
赛弗儿cypher 查询方法 逻辑也不同

疾病/症状：更直接的实体识别
功效：需要理解功能性描述
不同的提示词工程策略
'''

# 功效识别 抽取功效实体列表
class EffectCheckerAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        """判断用户输入是否涉及功效，并提取功效实体"""
        parser = PydanticOutputParser(pydantic_object=EffectCheckOutput)
        # 定义输出格式 get_format_instructions 方法返回一个字符串，描述了输出的格式。
        format_instructions = parser.get_format_instructions()

        # 获取 session_id
        session_id = agent_state.get("session_id", "default")
        # 拿到最近 k 条历史
        past_messages = get_top_k_history(session_id)
        # 把 提示词 和历史记录 与用户输入 组合成一个消息列表
        # 告诉 AI 角色：中医功效判断助手
        # 告诉 AI 任务：判断 + 提取功效
        # 带上历史对话
        # 带上用户当前问题
        messages = [
                       SystemMessage(content=(
                           "你是一个中医问句分析助手，任务有两部分：\n"
                           "1. 判断用户的问题是否涉及中药“功效”（如清热、补气、活血、镇静、止咳等）。只要问题涉及任何功效类表达，即算相关；否则为不相关。\n"
                           "2. 如果相关，请抽取其中所有功效关键词，返回一个功效列表。\n\n"
                           f"请严格按以下格式返回：\n{format_instructions}"
                       ))] + past_messages + [
                       HumanMessage(content=agent_state['input'])
                   ]

        raw_output = my_llm.invoke(messages).content.strip()
        # 解析输出 parser 用EffectCheckOutput模型解析输出结果
        parsed_output = parser.parse(raw_output)
        # 存储解析结果 is_effect_question 是否涉及功效，effect_entities 涉及功效的实体  到 agent_state
        agent_state['is_effect_question'] = parsed_output.is_effect_question
        agent_state['effect_entities'] = parsed_output.entities
        print(agent_state)
        return agent_state
