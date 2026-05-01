# File: MyAgents/EffectEntityEmbeddingMatcherAgent.py
import os
from langchain_core.runnables import Runnable
import faiss
import pickle
import numpy as np

from __004__langgraph_agent.agent_state import AgentState
from common.embedding_model import embedding_model
from common.config import Config

conf = Config()

''' 5 
查询faiss数据库 选取 最相似的实体 返回 结果 name 症状  类型 type 相似度 similarity
# index 索引路径
# RECORDS_PATH 节点记录路径
Pickle 文件（.pkl）：存原始文本 / 数据（比如 “感冒”“咳嗽”“清热解毒”）
FAISS 索引文件（.index）：存向量 + 检索结构（用来做快速语义搜索）查找原始数据的索引
读取索引文件和节点记录文件
'''
INDEX_PATH = os.path.join(conf.FAISS_INDEX_BASE, "tcm_effect_node_index.index")
RECORDS_PATH = os.path.join(conf.FAISS_INDEX_BASE, "tcm_effect_node_texts.pkl")

index = faiss.read_index(INDEX_PATH)
with open(RECORDS_PATH, "rb") as f:
    node_records = pickle.load(f) # 把数据读取到 node_records 内存中

# 是为了确保输入的向量都是 float32 类型，FAISS 只认 float32 类型的向量！
def _to_float32(a):
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return a

# _distances_to_similarity 转换 FAISS 返回的距离为“相似度分数”（越大越相似）
# 判断三种方法的相似度计算公式
# 如果是 l2 距离 返回-dists，
# 如果 l2 向量单位化，返回余弦相似度 dist
# 如果就是 ip 距离，直接返回 dist
def _distances_to_similarity(dists: np.ndarray, metric_type: int, assume_unit_norm: bool = True) -> np.ndarray:
    """
    将 FAISS 返回的距离转换为“相似度分数”（越大越相似）。
    - 对 METRIC_INNER_PRODUCT：直接返回 余弦相似度 dist（即内积≈cosine）
    - 对 METRIC_L2：若向量单位化，cos = 1 - L2/2；否则无法直接换算（退化成负的 L2）
    """
    dists = np.asarray(dists)
    # 对 METRIC_INNER_PRODUCT：直接返回 dists（即内积≈cosine）
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        # 返回余弦相似度 dist
        return dists
    # 对 METRIC_L2：若向量单位化，cos = 1 - L2/2；否则无法直接换算（退化成负的 L2）
    # 如果 faiss 使用的是 L2 距离，且向量单位化，返回余弦相似度 dist
    # 因为 L2是欧式距离 数据之间的距离越小，相似度越高 所以要减去距离再除以 2
    # 为什么除2：因为 L2 距离的范围是 [0, ∞)，而余弦相似度的范围是 [-1, 1]，
    elif metric_type == faiss.METRIC_L2:
        # 如果向量单位化，返回余弦相似度 dist
        # 向量单位化 是指将向量的长度归一化到 1，保持向量的方向
        if assume_unit_norm:
            return 1.0 - dists / 2.0  # map L2 to cosine similarity
        else:
            # 无法可靠换算，返回负的距离作为“相似度”的代理（仍保证越大越相似）
            return -dists
    else:
        # 其他度量很少用，退化处理
        return -dists

# 输入一批实体 → 去 FAISS 检索中药功效节点 → 返回高分匹配结果
# 拿到阈值>=0.85 的前 3 个结果
class EffectEntityEmbeddingMatcherAgent(Runnable):
    threshold = 0.85 #
    # 只返回相似度 ≥ 阈值的结果
    top_k = 3
    # 取 top_k 个候选

    # 匹配逻辑
    # 输入一批实体 → 去 FAISS 检索中药功效节点 → 返回高分匹配结果

    def match_entities(self, entities: list[str], debug: bool = False) -> list[dict]:
        """对输入的实体进行向量化匹配，只返回相似度 ≥ 阈值的结果"""
        # 如果输入list[str]为空，直接返回空列表
        if not entities:
            return []
        # 如果索引list[str]为空，直接返回空列表
        if index.ntotal == 0:
            if debug:
                print("[DEBUG] Index is empty (ntotal=0).")
            return []

        # 编码并确保 float32
        # entities 是输入 把数据进行embedding处理
        # normalize_embeddings=True 是指将向量的长度归一化到 1，保持向量的方向
        embeddings = embedding_model.encode(entities, normalize_embeddings=True)
        embeddings = _to_float32(embeddings)

        # indices：最相似的前 3 个节点的编号
        # distances：对应的距离值
        distances, indices = index.search(embeddings, self.top_k)

        # 将距离统一转为“相似度”
        # 自动判断索引是内积还是 L2
        # 把距离转成 0~1 之间的相似度
        # 去 FAISS 索引对象里，读取它自带的 metric_type 属性
        # 如果你用 IndexFlatIP 建的 → 它自带属性就是 METRIC_INNER_PRODUCT
        # 如果你用 IndexFlatL2 建的 → 它自带属性就是 METRIC_L2
        metric_type = getattr(index, "metric_type", faiss.METRIC_L2)
        # 用余弦相似度 来对比结果 （越大越相似）
        sims = _distances_to_similarity(distances, metric_type, assume_unit_norm=True)
        # match_results 存储匹配结果
        match_results = []

        # entities 是输入的实体 indices 是 top3 候选个节点的编号 sims 是 top3 sims候选的相似度
        # entity 是当前输入实体 idxs 是当前实体的 top3 候选节点编号 sim_row 是当前实体的 top3 候选相似度
        for e_idx, (entity, idxs, sim_row) in enumerate(zip(entities, indices, sims)):
            # 可选：调试输出查看原始距离与换算后的相似度
            # faiss_idx 是当前实体的 top3 候选节点编号
            # raw_d 是当前实体的 top3 候选原始距离
            # sim_row 是当前实体的 top3 候选相似度
            # name 是当前实体的 top3 候选真实名称
            # node_records[i][0]  是当前实体的 top3 候选真实名称
            # if i >= 0 else None 如果 i < 0，说明没有对应的节点，返回 None
            # 如果 debug 为 True，打印调试信息
            if debug:
                raw_d = distances[e_idx]
                dbg = [
                    {
                        "faiss_idx": int(i),
                        "raw_dist": float(rd),
                        "sim": float(sr),
                        "name": node_records[i][0] if i >= 0 else None
                    }
                    # 遍历当前实体的 top3 候选节点编号、原始距离、相似度
                    for i, rd, sr in zip(idxs, raw_d, sim_row)
                ]
                print(f"[DEBUG] entity='{entity}': top{self.top_k} raw={dbg}")
            # 如果 debug 为 False，遍历当前实体的 top3 候选节点编号、相似度
            for i, sim in zip(idxs, sim_row):
                if i < 0:  # faiss 可能返回 -1（不够样本）
                    continue
                if sim >= self.threshold:
                    name, label = node_records[i]
                    match_results.append({
                        "input_entity": entity,
                        "matched_entity": name,
                        "type": label,
                        "similarity": float(sim)
                    })
                '''
                type（类型）值
                Symptom（症状）
                Disease（疾病）
                Effect（功效）
                Herb（药材）
                Formula（方剂） 等等
      {"input_entity": "咳嗽", "matched_entity": "咳嗽", "type": "Symptom", "similarity": 0.975},
      {"input_entity": "咳嗽", "matched_entity": "干咳", "type": "Symptom", "similarity": 0.895}
                '''
        return match_results



    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        # 1. 从全局状态里取出要匹配的“功效实体”
        entities = agent_state.get('effect_entities', [])

        # 2. 调用你之前写的匹配逻辑（去FAISS搜索）
        match_results = self.match_entities(entities)

        # 3. 把匹配结果放回全局状态 effect_entity_match_results
        agent_state['effect_entity_match_results'] = match_results

        # 4. 打印日志
        print(f"功效匹配结果: {match_results}")

        # 5. 返回更新后的状态
        return agent_state


if __name__ == '__main__':
    agent = EffectEntityEmbeddingMatcherAgent()
    # 打开 debug 观察“清热”的 topK 候选的原始距离和相似度换算
    print(agent.match_entities(["清热"], debug=True))
