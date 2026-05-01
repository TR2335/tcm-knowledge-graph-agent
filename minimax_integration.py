"""
MiniMax-M2.5-highspeed API 集成示例
演示如何在 LangChain 中使用 MiniMax 作为 ChatOpenAI 后端

使用方式：
1. 安装依赖：pip install langchain-openai langchain-core
2. 设置环境变量：
   export MINIMAX_API_KEY="your_api_key"
   export MINIMAX_BASE_URL="https://api.minimax.chat/v1"
   export MINIMAX_MODEL="MiniMax-M2.5-highspeed"
3. 运行：python minimax_integration.py
"""

import os
from langchain_openai import ChatOpenAI

# MiniMax API 配置
llm = ChatOpenAI(
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
    api_key=os.getenv("MINIMAX_API_KEY"),
    model=os.getenv("MINIMAX_MODEL", "MiniMax-M2.5-highspeed"),
    temperature=0.7,
    max_tokens=2048,
)

# 简单对话测试
response = llm.invoke("请用30字介绍中医知识图谱的作用")
print("MiniMax 回复:", response.content)

# TCM问题分类示例（对应项目中的 TCMQuestionClassifierAgent）
system_prompt = """你是一个中医问题分类专家。
判断用户输入的问题是否属于中医范畴（包含中药、方剂、疾病、症状、功效等）。
回复格式：JSON，{"is_tcm": true/false, "confidence": 0.0-1.0, "reason": "分类理由"}"""

result = llm.invoke([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "我最近咳嗽发热，有什么中药可以调理？"}
])
print("\n问题分类结果:", result.content)

# Cypher查询生成示例（对应项目中的 CypherQueryGeneratorAgent）
cypher_prompt = """给定以下中医实体，从Neo4j知识图谱中查询相关信息。
实体：咳嗽、发热
关系类型：治疗、缓解、导致
请生成Cypher查询语句。"""

cypher_result = llm.invoke([
    {"role": "system", "content": "你是一个Neo4j Cypher查询专家，擅长生成准确的图数据库查询语句。"},
    {"role": "user", "content": cypher_prompt}
])
print("\nCypher生成结果:", cypher_result.content)
