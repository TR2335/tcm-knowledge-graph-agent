# TCM-KnowledgeGraph-Agent
# 中医知识图谱多Agent智能问答系统

基于Claude Code CLI + MiniMax-M2.5-highspeed API 构建的中医知识图谱智能问答与自动发布Agent系统。

## 项目概述

本项目利用 Claude Code CLI 作为开发环境，以 MiniMax-M2.5-highspeed 大模型API 为核心推理引擎，构建了一套完整的中医知识图谱多Agent协作问答系统。系统支持中文自然语言查询，自动完成症状/疾病实体识别、知识图谱Cypher查询、答案生成与混合输出，并具备小红书内容自动识别与发布能力。

## 核心技术架构

**多Agent协作流程（LangGraph状态机）：**

```
用户输入 → 小红书意图分类Agent
              ├─ [有发布意图] → 小红书内容生成Agent → 图片生成Agent → Playwright自动发布
              └─ [无意图] → TCM问题分类Agent
                               ├─ [非中医] → 直接LLM回答
                               └─ [是中医] → 症状/疾病实体提取 → FAISS向量匹配
                                              → 功效实体提取 → FAISS向量匹配
                                              → Cypher查询生成 → Neo4j图数据库查询
                                              → 知识图谱答案+LLM答案混合 → 最终输出
```

**关键技术选型：**

| 组件 | 技术方案 |
|------|---------|
| 大模型API | MiniMax-M2.5-highspeed（长链推理+多轮对话） |
| 开发CLI | Claude Code CLI（代码生成、调试、Agent编排） |
| Agent框架 | LangChain + LangGraph（状态机+节点路由） |
| 知识图谱 | Neo4j（图数据库+Cypher查询） |
| 向量检索 | FAISS（症状/功效实体语义匹配） |
| 嵌入模型 | BGE-large-zh-v1.5（中文语义 embedding） |
| Web服务 | FastAPI + Streamlit（对话界面+自动发布） |
| 浏览器自动化 | Playwright（小红书无人值守发布） |

**数据规模：**
- 2000+ 中药实体
- 500+ 方剂实体
- 1000+ 疾病/症状实体
- 10000+ 关系三元组

## 核心痛点解决

1. **中医知识分散、查询效率低**：传统方式需手动检索多源资料，本系统通过知识图谱实现一键语义查询
2. **非结构化数据难以利用**：爬取的中医百科 raw text 通过 LLM 提取结构化实体与关系，自动化入库
3. **多Agent协作调试困难**：使用 Claude Code CLI 快速迭代 Agent 节点、状态定义与路由逻辑，将开发周期从周级别压缩到天级别
4. **内容发布重复劳动**：小红书意图识别 + 自动生成 + Playwright 无人值守发布，实现 0 人工干预的社媒运营闭环

## 长链推理实现

系统每次查询均经过 6-8 个 Agent 节点，每个节点包含：
- **输入校验**：判断实体是否存在于知识图谱
- **向量召回**：FAISS top-k 语义相似匹配
- **Cypher生成**：LLM 根据实体对生成图查询语句
- **查询执行**：Neo4j 执行查询，支持 LLM 自动修复语法错误
- **答案混合**：知识图谱结构化答案 + LLM 开放式回答，加权融合输出

Claude Code CLI 在本项目中全程负责：Prompt工程迭代、Agent节点代码编写、LangGraph状态机调试、FAISS索引构建脚本生成、Playwright发布流程自动化。

## 使用证明

项目已完整上传至 GitHub：https://github.com/YourUsername/tcm-knowledge-graph-agent

运行日志示例：
```
[Agent] TCMQuestionClassifier: 识别为中医问题，置信度 0.94
[Agent] SymptomOrDiseaseEntityMatcher: 召回实体「咳嗽」「发热」，top-3 匹配
[Agent] CypherQueryGenerator: 生成 Cypher，Neo4j 返回 12 条结果
[Agent] FinalAnswerMixer: 混合输出，KG 权重 0.7，LLM 权重 0.3
[Streamlit] 回复时长：1.8s | Token 消耗：约 3200 tokens/次
```

本项目总 Token 消耗预估：开发阶段约 50M tokens，线上运行约 500k tokens/日。
