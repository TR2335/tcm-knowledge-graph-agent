from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage
import json
import os

from __004__langgraph_agent.agent_state import AgentState
from common.llm import my_llm
from common.neo4j_client import neo4j_client
from common.config import Config

conf = Config()
'''
根据用户问题和实体信息，生成一条合适的 Cypher 查询语句。
主要方法 FAISS_INDEX_BASE 是图谱的索引路径，用于存储和检索实体信息。
通过union 并多个match 查询 return 的字段一致 只返回 name 或 result 字段。
UNION 是 Neo4j Cypher 赛弗儿 的结果合并关键字，作用是把多个独立 MATCH 查询的结果，上下堆叠成一个统一结果集neo4j.com。
当回答作用类问题时，治疗疾病，功效，缓解症状，都是其作用。
把具体图节点和关系类型也加入到 prompt 中，帮助模型理解上下文。
把格式要求也加入到 prompt 中，帮助模型生成符合要求的查询语句。

从状态当中获取用户问题 
用 faiss 查询 结果 包含症状/疾病/功效 等实体 然后
构建提示词 定义角色 生成 cyhher 查询  如果无法生成 则返回空字符串 给出cot链
用union合并多个match查询 固定输入输出格式 

把提示词 + faiss 查询结果 + 用户问题 作为输入 给模型 生成 Cypher 查询语句
然后 执行 Cypher 查询 得到查询结果 通过模型 转换为自然语言回答

'''

# 打开图谱结构文件 6
# 读取中医知识图谱的所有节点类型、关系类型
# 存入 schema_str，给 AI 看图谱长什么样
with open(os.path.join(conf.FAISS_INDEX_BASE, "tcm_metadata.json"), "r", encoding="utf-8") as f:
    schema_str = f.read()

SYSTEM_PROMPT = """
                    你是一个中医图谱查询专家，你能根据用户的问题，参考以下图数据库结构，生成一条合适的 Cypher 查询语句（使用 Cypher 语法），
                    并给出简要的查询意图说明。
                    如果问题无法用图谱进行回答，那么就返回空字符串。
                    如果使用 UNION 合并多个 MATCH 查询，所有 RETURN 的字段名必须一致，建议统一为 name 或 result。
                    当回答作用类问题时，治疗疾病，功效，缓解症状，都是其作用。
                    图谱节点和关系类型如下：
                    {schema}
                    
                    请严格按照如下格式输出：
                    Cypher:
                    <cypher语句>
                    
                    Explanation:
                    <语句用途说明>
                    
                    读取图谱结构
                    接收用户问题 + 识别出的实体
                    AI 自动生成 Cypher 查询语句
                    语法检查、执行查询
                    AI 把图谱结果转成自然语言回答
                    返回最终答案
                    """
#todo 你是中医图谱专家，生成 Cypher 查询
# 不会答就返回空字符串
# 多查询合并必须统一返回字段用union 合并多个 MATCH 查询的结果（name/result  ruizate）
# 治疗、功效、缓解症状都算 “作用”
# 提供图谱结构

# 把 图谱数据 变成schema 加入到 prompt 中，帮助模型理解上下文 把真实的图谱结构填充进提示词，让 AI 知道数据库里有什么。
system_prompt = SYSTEM_PROMPT.format(schema=schema_str)

# 生成 Cypher 查询语句
# 接收问题 → 生成 Cypher → 查询图谱 → 返回答案。
class CypherQueryGeneratorAgent(Runnable):
    #todo 调用 LLM 生成 Cypher 查询语句
    # 输入：agent_state（包含问题、识别出的实体等所有信息）
    # 输出：更新后的状态（包含 Cypher、查询结果、答案）
    # 功能：根据自然语言与实体信息生成 Cypher 查询并执行
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        # 从状态中取出用户原始问题。 获取用户输入的问题
        user_question = agent_state['input']
        '''
        提取 faiss 结果 
        
        '''
        # 提取   识别出的症状 / 疾病列表
        matched_symptoms = [m['matched_entity'] for m in agent_state.get('symptom_or_disease_entity_match_results', [])]
        # 提取识别出的功效列表。
        matched_effects = [m['matched_entity'] for m in agent_state.get('effect_entity_match_results', [])]

        context_lines = []
        # 构造提示词
        # 如果有症状 / 疾病，加入提示词。
        if matched_symptoms:
            context_lines.append(f"用户提到的症状包括：{', '.join(matched_symptoms)}。")
        # 如果有功效，加入提示词。
        if matched_effects:
            context_lines.append(f"用户提到的功效包括：{', '.join(matched_effects)}。")
        entity_context = "\n".join(context_lines)

        # 构造 prompt 把 提示词 + faiss + neo4j 和原始问题 加入到 prompt 中 用来生成查询
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{entity_context}\n\n原始问题：{user_question}")
        ]

        print("🧠 正在调用 LLM 生成 Cypher 查询...")
        # 调用 LLM 生成 Cypher 查询语句
        response = my_llm.invoke(messages).content

        try:
            '''
            模型生成的 Cy4j 查询语句：
            Cypher:
            MATCH (f:方剂)-[:主治]->(d:疾病{name:"感冒"}) RETURN f.name AS result
            
            Explanation:
            查询能够治疗感冒的所有方剂名称
            '''
            # 去掉Cypher:和Explanation:的空格
            cypher = response.split("Cypher:")[1].split("Explanation:")[0].strip()
            # 从 Explanation: 切开
            # 拿后面的解释文字
            # 去掉空格换行
            explanation = response.split("Explanation:")[1].strip()
        except Exception as e:
            # 空 Cypher
            # 空结果
            # 提示无法查询
            # 直接返回状态
            agent_state['cypher'] = ""
            agent_state['cypher_result'] = []
            agent_state['cypher_answer'] = "此问题不适合用图谱查询。"
            return agent_state

        print("✅ 生成 Cypher：", cypher)

        # 用explain检查语法 防止执行错误的 Cypher 语句 防止注入攻击
        # ok 和 msg 分别表示是否成功和错误信息
        # msg 是错误信息
        ok, msg = neo4j_client.check_cypher_syntax(cypher)
        if not ok:
            agent_state['cypher'] = cypher
            agent_state['cypher_result'] = []
            agent_state['cypher_answer'] = f"Cypher 查询语法错误：{msg}"
            return agent_state
        # 执行 Cypher 查询 得到查询结果
        result = neo4j_client.run_cypher(cypher)
        print("🔍 执行 Cypher 查询结果：", result)
        # 把 Cypher 语句和查询结果存入状态。
        agent_state['cypher'] = cypher
        agent_state['cypher_result'] = result # 查询结果

        # 专家身份
        # 输入：原问题 + 图谱结果 转换成自然语言回答
        summary_prompt = [
            SystemMessage(content="你是一个中医知识专家，请将以下图谱查询结果总结为通俗自然语言："),
            HumanMessage(
                content=f"原问题：{user_question}\n\n查询结果：\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        ]
        # 调用 AI，生成自然语言答案。
        summary_response = my_llm.invoke(summary_prompt).content
        agent_state['cypher_answer'] = summary_response
        agent_state['output'] = summary_response
        # 把最终答案存入状态
        print("📢 生成的 cypher_answer：", summary_response)
        print("🔚 AgentState:", agent_state)

        return agent_state

    '''
    读取图谱结构
接收用户问题 + 识别出的实体
AI 自动生成 Cypher 查询语句
语法检查、执行查询
AI 把图谱结果转成自然语言回答
返回最终答案
    '''
