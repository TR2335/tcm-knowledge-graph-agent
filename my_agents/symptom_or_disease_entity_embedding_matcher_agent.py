# File: MyAgents/SymptomOrDiseaseEntityEmbeddingMatcherAgent.py
import os
from langchain_core.runnables import Runnable
import faiss
import pickle
import numpy as np

from __004__langgraph_agent.agent_state import AgentState
from common.embedding_model import embedding_model
from common.config import Config

conf = Config()

# 初始化嵌入模型和索引 3
# INDEX_PATH：FAISS 向量索引文件（存向量）
# RECORDS_PATH：对应的疾病 / 症状名称和类型（存文字）
INDEX_PATH = os.path.join(conf.FAISS_INDEX_BASE, "tcm_disease_symptom_node_index.index")
RECORDS_PATH = os.path.join(conf.FAISS_INDEX_BASE, "tcm_disease_symptom_node_texts.pkl")


# 把硬盘里的向量库加载到内存
# 作用：准备好做向量搜索
index = faiss.read_index(INDEX_PATH)


# 读取向量对应的症状 / 疾病名称 + 类型
# node_records 格式大概是：[("咳嗽", "symptom"), ("感冒", "disease"), ...]
with open(RECORDS_PATH, "rb") as f:
    node_records = pickle.load(f)

# 确保向量是 float32 类型 faiss只能处理 float32 类型的向量
def _to_float32(a):
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return a


def _distances_to_similarity(dists: np.ndarray, metric_type: int, assume_unit_norm: bool = True) -> np.ndarray:
    """
    将 FAISS 返回的距离转换为“相似度分数”（越大越相似）。
    - METRIC_INNER_PRODUCT：返回原值（在归一化前提下≈cosine）
    - METRIC_L2：若单位化，cos = 1 - L2/2；否则退化为 -L2 以保持“越大越相似”
    """
    #todo
    # FAISS 返回的是距离（越小越相似）
    # 程序需要相似度（越大越相似）
    # 这个函数负责换算
    # metric_type 是 FAISS 索引的指标类型，默认是 L2
    # dists把 FAISS 返回的距离转成数组，方便计算。
    dists = np.asarray(dists)
    #todo
    # 如果用的是 METRIC_INNER_PRODUCT 内积（点积） 方式计算相似度
    # 它本来就是 “越大越像”
    # 所以不用改，直接返回。
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return dists
    #todo
    # 如果用的是 L2 距离（直线距离）
    # 这个是 越小越像，必须转换！
    elif metric_type == faiss.METRIC_L2:
        if assume_unit_norm:
            return 1.0 - dists / 2.0
        #todo
        # 如果向量已经单位化（归一化）了
        # 我们可以用公式把 L2 距离 → 变成 0~1 之间的余弦相似度
        else:
            return -dists
    else:
        return -dists

# 类名：症状 / 疾病向量匹配智能体
# 继承 Runnable → 可放入工作流执行
class SymptomOrDiseaseEntityEmbeddingMatcherAgent(Runnable):
    # 允许通过 Config 覆盖；没有就用默认
    threshold = 0.85
    top_k = 3
    # threshold = 0.85：相似度大于等于 0.85 才算匹配成功（太低会乱匹配）
    # top_k = 3：每个词最多找 3 个最相似的

    # 输入：用户说的词列表 ["咳嗽", "头痛"]
    # 输出：匹配到的标准中医词 + 类型 + 相似度
    def match_entities(self, entities: list[str], debug: bool = False) -> list[dict]:
        """
        对 symptom_or_disease 实体进行向量化匹配，只返回相似度 ≥ 阈值的结果。
        返回项包含：input_entity / matched_entity / type / similarity
        """
        # 如果没有实体，直接返回空
        if not entities:
            return []
        # 如果向量库为空，直接返回空
        if index.ntotal == 0:
            if debug:
                print("[DEBUG] Index is empty (ntotal=0).")
            return []

        # 编码并确保 float32
        # 用三元组进行搜索 就可以防止提示词注入问题 也解决了 embedding 收集提问 与 搜索时的不一致问题
        embeddings = embedding_model.encode(entities, normalize_embeddings=True)
        embeddings = _to_float32(embeddings)

        # 搜索 获取每个实体的 top_k=3个最相似的索引 distances是 距离数组 indices是 索引数组
        distances, indices = index.search(embeddings, self.top_k)

        # getattr用 getattr 就绝对不会报错，拿不到就自动用默认值。
        metric_type = getattr(index, "metric_type", faiss.METRIC_L2)
        # 用 _distances_to_similarity 函数 把距离 → 变成 0~1 的相似度
        sims = _distances_to_similarity(distances, metric_type, assume_unit_norm=True)

        # 遍历每个用户输入的实体，获取 top_k=3个最相似的索引和相似度
        match_results = []
        for row_idx, (entity, idxs, sim_row) in enumerate(zip(entities, indices, sims)):
        #todo 遍历用户输入的每一个词（比如 “咳嗽”“头痛”）
        # 每个词对应：
        # entity：用户输入的词
        # idxs：FAISS 找到的 3 个最相似的编号
        # sim_row：这 3 个编号对应的相似度（0~1）
            if debug:
                # 如果 debug=True，打印每个词的原始距离、相似度、匹配名称
                raw_d = distances[row_idx]
                # 数组距离
                dbg = [
                    {
                        "faiss_idx": int(i), # FAISS 内部编号
                        "raw_dist": float(rd), # 原始距离
                        "sim": float(sr),   # 转换后的相似度
                        "name": node_records[i][0] if i >= 0 else None # 匹配到的名称
                    }
                    # 循环每个词的 top_k=3 个最相似的编号
                    for i, rd, sr in zip(idxs, raw_d, sim_row)
                ]
                #todo 调试用的打印
                # 把每个词的：
                # FAISS 内部编号
                # 原始距离
                # 转换后的相似度
                # 匹配到的名称
                # 全部打印出来给你看，方便查问题
                print(f"[DEBUG] entity='{entity}': top{self.top_k} raw={dbg}")
            # 遍历每个词的 top_k=3 个最相似的编号
            for i, sim in zip(idxs, sim_row):
                # FAISS 找不到时会返回 -1
                # 无效编号，直接跳过
                if i < 0:
                    continue
                # 只有相似度 ≥ 0.85 才算有效匹配
                # 低于 0.85 直接丢掉，避免乱匹配
                if sim >= self.threshold:
                    name, label = node_records[i]
                    match_results.append({
                        "input_entity": entity, # # 用户说的 如 "咳嗽"
                        "matched_entity": name, # 匹配到的标准中医词 如 "咳嗽"
                        "type": label, # 类型（症状 / 疾病）
                        "similarity": float(sim) # 相似度（0~1）
                    })
                # 从知识库中拿出：
                # name：标准名称（比如 “咳嗽”）
                # label：类型（症状 / 疾病）
        return match_results

    # 这是 LangGraph 工作流节点的入口方法
    # 外界调用这个节点，就会跑这里
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        # 从对话状态里拿到上一步提取的症状 / 疾病词
        # 例如：["咳嗽", "头痛"]
        """对 symptom_or_disease_entities 进行向量化匹配，返回匹配结果到 agent_state"""
        entities = agent_state.get('symptom_or_disease_entities', [])
        # 调用上面那段核心代码
        # 做向量匹配 → 得到标准词
        match_results = self.match_entities(entities)
        agent_state['symptom_or_disease_entity_match_results'] = match_results
        print(f"匹配结果: {match_results}")
        return agent_state

    # 将用户输入向量化
    # 把 FAISS 的距离 → 转成 0~1 相似度
    # 遍历用户输入的每个词，只保留相似度 ≥0.85 的标准中医词
    # 把匹配结果存到对话状态，给后面流程使用
if __name__ == '__main__':
    agent = SymptomOrDiseaseEntityEmbeddingMatcherAgent()
    # 调试查看“清热”或任意输入的 topK 候选、原始距离与相似度换算
    print(agent.match_entities(["咳嗽"], debug=True))