# ============================================================
# LangGraph 状态图构建文件
# 作用：定义整个多 Agent 问答系统的工作流编排
# ============================================================

# ---------------------- 导入相关模块 ----------------------
from langgraph.graph import StateGraph, END  # END 是终止节点标记

# 导入各个 Agent（每个 Agent 负责一个特定任务）
from __004__langgraph_agent.my_agents.cypher_query_generator_agent import CypherQueryGeneratorAgent  # 生成 Neo4j 查询
from __004__langgraph_agent.my_agents.direct_llm_answer_agent import DirectLLMAnswerAgent  # LLM 直接回答（兜底）
from __004__langgraph_agent.my_agents.effect_checker_agent import EffectCheckerAgent  # 检查功效实体
from __004__langgraph_agent.my_agents.effect_entity_embedding_matcher_agent import EffectEntityEmbeddingMatcherAgent  # FAISS 功效匹配
from __004__langgraph_agent.my_agents.final_answer_mixer_agent import FinalAnswerMixerAgent  # 融合两个答案
from __004__langgraph_agent.my_agents.symptom_or_disease_checker_agent import SymptomOrDiseaseCheckerAgent  # 检查症状/疾病实体
from __004__langgraph_agent.my_agents.symptom_or_disease_entity_embedding_matcher_agent import \
    SymptomOrDiseaseEntityEmbeddingMatcherAgent  # FAISS 症状/疾病匹配
from __004__langgraph_agent.my_agents.tcm_question_classifier_agent import TCMQuestionClassifierAgent  # 判断是否中医问题
from __004__langgraph_agent.my_agents.xiaohongshu_image_generator_agent import XiaohongshuImageGeneratorAgent  # 小红书图片生成
from __004__langgraph_agent.my_agents.xiaohongshu_intent_classifier_agent import XiaohongshuIntentClassifierAgent  # 小红书意图识别
from __004__langgraph_agent.my_agents.xiaohongshu_tcm_post_output_agent import XiaohongshuTCMPostAgent  # 小红书内容生成
from __004__langgraph_agent.agent_state import AgentState  # 状态数据结构定义
from common.llm import get_session_history  # 获取对话历史


# ============================================================
# 第一步：构建状态图
# StateGraph 是 LangGraph 的核心，用来实现多 Agent 协作工作流
# ============================================================
graph = StateGraph(AgentState)

# ============================================================
# 第二步：向状态图中添加各个 Agent 节点
# 每个节点就是一个 Agent，节点名用 Agent 类的 __name__（类名）
# ============================================================

# 小红书意图分类节点
graph.add_node(XiaohongshuIntentClassifierAgent.__name__, XiaohongshuIntentClassifierAgent())
# 小红书内容生成节点
graph.add_node(XiaohongshuTCMPostAgent.__name__, XiaohongshuTCMPostAgent())
# 小红书图片生成节点
graph.add_node(XiaohongshuImageGeneratorAgent.__name__, XiaohongshuImageGeneratorAgent())
# 中医问题分类节点（判断是否中医问题）
graph.add_node(TCMQuestionClassifierAgent.__name__, TCMQuestionClassifierAgent())
# LLM 直接回答节点（图谱答不了时的兜底）
graph.add_node(DirectLLMAnswerAgent.__name__, DirectLLMAnswerAgent())
# 症状/疾病提取节点
graph.add_node(SymptomOrDiseaseCheckerAgent.__name__, SymptomOrDiseaseCheckerAgent())
# 症状/疾病向量匹配节点（FAISS）
graph.add_node(SymptomOrDiseaseEntityEmbeddingMatcherAgent.__name__, SymptomOrDiseaseEntityEmbeddingMatcherAgent())
# 功效提取节点
graph.add_node(EffectCheckerAgent.__name__, EffectCheckerAgent())
# 功效向量匹配节点（FAISS）
graph.add_node(EffectEntityEmbeddingMatcherAgent.__name__, EffectEntityEmbeddingMatcherAgent())
# Cypher 查询生成节点（连接 FAISS 结果与 Neo4j）
graph.add_node(CypherQueryGeneratorAgent.__name__, CypherQueryGeneratorAgent())
# 最终答案融合节点（合并图谱答案 + LLM 答案）
graph.add_node(FinalAnswerMixerAgent.__name__, FinalAnswerMixerAgent())


# ============================================================
# 第三步：设置工作流入口节点
# 用户输入首先进入 XiaohongshuIntentClassifierAgent 进行意图识别
# ============================================================
graph.set_entry_point(XiaohongshuIntentClassifierAgent.__name__)


# ============================================================
# 第四步：定义条件路由函数
# 条件边：根据 AgentState 中的状态字段，决定下一步走哪个分支
# ============================================================

# ----- 小红书意图分类的路由 -----
def xiaohongshu_intent_classifier_agent_routing_control(state_agent: AgentState):
    """
    判断是否有小红书创作意图
    - 有小红书意图 → 生成小红书内容
    - 无小红书意图 → 进入中医问答流程
    """
    if state_agent['is_has_xhs_intent']:
        return XiaohongshuTCMPostAgent.__name__
    else:
        return TCMQuestionClassifierAgent.__name__


# ----- 添加条件边：小红书意图分类 → 根据结果分流 -----
graph.add_conditional_edges(
    XiaohongshuIntentClassifierAgent.__name__,
    xiaohongshu_intent_classifier_agent_routing_control
)

# ----- 小红书内容生成完成后，直接进入图片生成（无条件边） -----
graph.add_edge(XiaohongshuTCMPostAgent.__name__, XiaohongshuImageGeneratorAgent.__name__)


# ----- 中医问题分类的路由 -----
def tcm_question_classifier_agent_routing_control(state_agent: AgentState):
    """
    判断用户问题是否属于中医相关
    - 是中医问题 → 进入症状/疾病提取流程
    - 非中医问题 → LLM 直接回答后结束
    """
    if state_agent['is_tcm_question']:
        return SymptomOrDiseaseCheckerAgent.__name__
    else:
        return DirectLLMAnswerAgent.__name__


# ----- 添加条件边：中医问题分类 → 根据结果分流 -----
graph.add_conditional_edges(
    TCMQuestionClassifierAgent.__name__,
    tcm_question_classifier_agent_routing_control
)


# ----- 症状/疾病检查的路由 -----
def symptom_disease_checker_agent_routing_control(state_agent: AgentState):
    """
    判断是否提取到了症状/疾病实体
    - 有症状/疾病实体 → FAISS 向量匹配
    - 没有 → 跳过匹配，直接进入功效检查
    """
    if state_agent['is_symptom_or_disease'] and state_agent['symptom_or_disease_entities']:
        return SymptomOrDiseaseEntityEmbeddingMatcherAgent.__name__
    else:
        return EffectCheckerAgent.__name__


# ----- 添加条件边：症状/疾病检查 → 根据结果分流 -----
graph.add_conditional_edges(
    SymptomOrDiseaseCheckerAgent.__name__,
    symptom_disease_checker_agent_routing_control
)

# ----- 症状/疾病匹配完成后，无条件进入功效检查 -----
graph.add_edge(
    SymptomOrDiseaseEntityEmbeddingMatcherAgent.__name__,
    EffectCheckerAgent.__name__
)


# ----- 功效检查的路由 -----
def effect_checker_agent_routing_control(state_agent: AgentState):
    """
    判断是否提取到了功效实体
    - 有功效实体 → FAISS 向量匹配
    - 没有 → 直接进入 Cypher 查询生成
    """
    if state_agent['is_effect_question'] and state_agent['effect_entities']:
        return EffectEntityEmbeddingMatcherAgent.__name__
    else:
        return CypherQueryGeneratorAgent.__name__


# ----- 添加条件边：功效检查 → 根据结果分流 -----
graph.add_conditional_edges(
    EffectCheckerAgent.__name__,
    effect_checker_agent_routing_control
)

# ----- 功效匹配完成后，无条件进入 Cypher 查询生成 -----
graph.add_edge(
    EffectEntityEmbeddingMatcherAgent.__name__,
    CypherQueryGeneratorAgent.__name__
)

# ----- Cypher 查询生成完成后，无条件进入 LLM 直接回答（兜底） -----
graph.add_edge(
    CypherQueryGeneratorAgent.__name__,
    DirectLLMAnswerAgent.__name__
)


# ----- LLM 直接回答后的路由 -----
def direct_answer_agent_routing_control(state_agent: AgentState):
    """
    判断是否需要进入答案融合
    - 是中医问题（图谱可能回答了） → 进入答案融合
    - 非中医问题（已经直接回答了） → 结束
    """
    if state_agent['is_tcm_question']:
        return FinalAnswerMixerAgent.__name__
    else:
        return END


# ----- 添加条件边：LLM 回答 → 根据结果决定是否融合 -----
graph.add_conditional_edges(
    DirectLLMAnswerAgent.__name__,
    direct_answer_agent_routing_control
)

# ----- 最终答案融合完成后，结束工作流 -----
graph.add_edge(FinalAnswerMixerAgent.__name__, END)


# ============================================================
# 第五步：编译状态图
# compile() 将工作流编排编译成可执行的应用
# 之后通过 app.invoke() 调用
# ============================================================
app = graph.compile()


# ============================================================
# 第六步：定义外部调用入口
# 对外暴露的函数，接收用户输入，返回答案
# ============================================================
def call_langgraph_ai(user_id, session_id, user_content):
    """
    调用编译后的 LangGraph 应用处理用户问题

    参数:
        user_id: 用户唯一标识
        session_id: 会话 ID（用于历史记录）
        user_content: 用户输入的问题

    返回:
        tuple: (is_xhs_intent, content_tuple)
        - is_xhs_intent=False: 返回 (response_output,) 普通问答
        - is_xhs_intent=True: 返回 (title, content, image_path) 小红书内容
    """
    # 调用编译后的状态图应用，传入初始状态
    response = app.invoke({
        "input": user_content,
        "user_id": user_id,
        "session_id": session_id
    })

    # 根据是否有小红书意图，返回不同格式的结果
    if not response['is_has_xhs_intent']:
        # 普通问答：返回最终回答
        response_output = response['output']

        # 保存对话历史
        history = get_session_history(session_id)
        history.add_user_message(user_content)
        history.add_ai_message(response_output)

        return False, (response_output,)
    else:
        # 小红书创作：返回标题 + 内容 + 图片路径
        title = response['xiaohongshu_tcm_post_title']
        content = response['xiaohongshu_tcm_post_content']
        image_path = response['xiaohongshu_tcm_post_image_path']

        # 保存对话历史
        history = get_session_history(session_id)
        history.add_user_message(user_content)
        history.add_ai_message(title + content)

        return True, (title, content, image_path)


# ============================================================
# 附：旧版调用方法（已废弃，保留参考）
# ============================================================
# def call_my_ai(messages):
#     # 验证输入的消息是否有效
#     if messages is None or len(messages) == 0 or messages[-1]['role'] != 'user':
#         return ""
#     print(messages)
#     # 调用编译后的状态图应用
#     response = app.invoke({"input": messages[-1]['content']})
#     if not response['is_has_xhs_intent']:
#         return False, (response['output'],)
#     else:
#         return True, (
#             response['xiaohongshu_tcm_post_title'],
#             response['xiaohongshu_tcm_post_content'],
#             response['xiaohongshu_tcm_post_strategies'],
#             response['xiaohongshu_tcm_post_image_path']
#         )


# ============================================================
# 主函数入口（测试用）
# ============================================================
if __name__ == '__main__':
    # 测试问答功能
    call_langgraph_ai("user_001", "session_001", "咳嗽吃什么药呢？")
