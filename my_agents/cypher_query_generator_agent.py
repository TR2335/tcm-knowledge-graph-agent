from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
import json
import os
from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm
from common.neo4j_client import neo4j_client
from common.config import Config

conf = Config()

with open(os.path.join(conf.FAISS_INDEX_BASE, "tcm_metadata.json"), "r", encoding="utf-8") as f:
    schema_str = f.read()

SYSTEM_PROMPT = """你是一个中医图谱查询专家，根据以下图数据库结构生成 Cypher 查询语句。
如果问题无法用图谱回答，返回空字符串。
使用 UNION 合并多个 MATCH 查询时，所有 RETURN 字段名必须一致（建议 name 或 result）。
治疗疾病、功效、缓解症状都算"作用"。
图谱结构：{schema}
输出格式：
Cypher:
<cypher语句>
Explanation:
<语句用途说明>"""

system_prompt = SYSTEM_PROMPT.format(schema=schema_str)


class CypherQueryGeneratorAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        user_question = agent_state['input']
        matched_symptoms = [m['matched_entity'] for m in agent_state.get('symptom_or_disease_entity_match_results', [])]
        matched_effects = [m['matched_entity'] for m in agent_state.get('effect_entity_match_results', [])]
        context_lines = []
        if matched_symptoms:
            context_lines.append(f"用户提到的症状包括：{', '.join(matched_symptoms)}。")
        if matched_effects:
            context_lines.append(f"用户提到的功效包括：{', '.join(matched_effects)}。")
        entity_context = "\n".join(context_lines)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{entity_context}\n\n原始问题：{user_question}")
        ]
        response = my_llm.invoke(messages).content
        try:
            cypher = response.split("Cypher:")[1].split("Explanation:")[0].strip()
        except Exception:
            agent_state['cypher'] = ""
            agent_state['cypher_result'] = []
            agent_state['cypher_answer'] = "此问题不适合用图谱查询。"
            return agent_state
        ok, msg = neo4j_client.check_cypher_syntax(cypher)
        if not ok:
            agent_state['cypher'] = cypher
            agent_state['cypher_result'] = []
            agent_state['cypher_answer'] = f"Cypher 查询语法错误：{msg}"
            return agent_state
        result = neo4j_client.run_cypher(cypher)
        agent_state['cypher'] = cypher
        agent_state['cypher_result'] = result
        summary_prompt = [
            SystemMessage(content="你是一个中医知识专家，请将以下图谱查询结果总结为通俗自然语言："),
            HumanMessage(content=f"原问题：{user_question}\n\n查询结果：\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        ]
        summary_response = my_llm.invoke(summary_prompt).content
        agent_state['cypher_answer'] = summary_response
        agent_state['output'] = summary_response
        return agent_state
