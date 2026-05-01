from sentence_transformers import SentenceTransformer
import faiss
import pickle

from common.embedding_model import embedding_model
from common.neo4j_client import neo4j_client


# 疾病和症状 所有的节点和数据
# 定义获取文本嵌入的函数
def get_embedding(texts):
    # 使用模型生成向量，并进行归一化（推荐用于余弦/欧氏距离计算）
    return embedding_model.encode(texts, normalize_embeddings=True)


# 从图谱中提取所有症状和疾病
records = neo4j_client.get_all_disease_and_symptom_names()

texts = [f"{name}" for name, label in records]  # eg: Symptom:咳嗽
embeddings = get_embedding(texts)

# 构建向量索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 保存索引和映射
faiss.write_index(index, "tcm_disease_symptom_node_index.index")
with open("tcm_disease_symptom_node_texts.pkl", "wb") as f:
    pickle.dump(records, f)