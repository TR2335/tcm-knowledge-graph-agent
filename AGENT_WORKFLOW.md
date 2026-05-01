"""
Agent 工作流架构说明
=========================

本项目采用 LangGraph 状态机实现多 Agent 协作推理。

工作流节点定义（__004__langgraph_more_agent.py）：

节点1: XiaohongshuIntentClassifier
  输入: user_content（用户原始输入）
  判断: 是否包含小红书发布意图（关键词：发笔记、写文章、分享到小红书等）
  输出: is_has_xhs_intent (bool) → 决定主路由方向

节点2: XiaohongshuTCMPostAgent (条件分支)
  输入: user_content + session_id
  调用: MiniMax-M2.5-highspeed 生成中医科普文案
  输出: xhs_title, xhs_content

节点3: XiaohongshuImageGeneratorAgent
  输入: xhs_content
  调用: 即梦(JiMeng) 图片生成API
  输出: image_paths

节点4: PlaywrightAutoPublisher
  输入: title + content + image_paths
  操作: 加载浏览器cookie → 访问创作者中心 → 填表 → 上传图片 → 发布
  输出: publish_status

节点5: TCMQuestionClassifierAgent (主问答链路)
  输入: user_content
  判断: 是否属于中医问题
  输出: is_tcm (bool)

节点6: SymptomOrDiseaseCheckerAgent
  输入: user_content
  LLM提取: 症状/疾病实体列表
  输出: symptom_disease_entities[]

节点7: SymptomOrDiseaseEntityEmbeddingMatcherAgent
  输入: symptom_disease_entities
  操作: BGE-large-zh-v1.5 向量化 → FAISS top-k 匹配
  输出: matched_kg_entities[]

节点8: EffectCheckerAgent
  输入: user_content
  LLM提取: 功效实体列表（如：清热、解表、补气）
  输出: effect_entities[]

节点9: EffectEntityEmbeddingMatcherAgent
  输入: effect_entities
  操作: 同节点7，FAISS匹配
  输出: matched_effect_entities[]

节点10: CypherQueryGeneratorAgent
  输入: matched_kg_entities + matched_effect_entities
  LLM生成: Cypher查询语句（支持语法自动修复）
  操作: Neo4j执行查询
  输出: cypher_result (list)

节点11: DirectLLMAnswerAgent（兜底）
  输入: user_content + cypher_result（空时触发）
  调用: MiniMax 直接回答
  输出: llm_answer

节点12: FinalAnswerMixerAgent
  输入: cypher_result + llm_answer
  混合: 知识图谱答案(权重0.7) + LLM答案(权重0.3)
  输出: final_answer

状态传递（agent_state.py TypedDict）:
  user_content → is_has_xhs_intent → [分支]
  symptom_disease_entities → matched_kg_entities → cypher_query → cypher_result → final_answer
  effect_entities → matched_effect_entities
  xhs_title/content → xhs_image_urls → publish_status
"""

# 工作流路由伪代码
"""
if is_has_xhs_intent:
    xhs_content = xhs_post_agent(user_content)
    xhs_images = image_agent(xhs_content)
    publish_status = playwright_publish(xhs_content, xhs_images)
else:
    if is_tcm:
        entities = symptom_checker(user_content)
        matched = embedding_matcher(entities)
        cypher = cypher_generator(matched)
        kg_result = neo4j.execute(cypher)
        if not kg_result:
            llm_answer = direct_llm(user_content)
        else:
            final = mixer(kg_result, llm_answer)
    else:
        final = direct_llm(user_content)
"""
