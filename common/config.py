import os
from dotenv import load_dotenv

from tools.path_utils import resolve_from_project_root

load_dotenv(".env")
load_dotenv(resolve_from_project_root(".env"))

FAISS_INDEX_BASE = os.getenv("FAISS_INDEX_BASE", resolve_from_project_root("__004__langgraph_agent"))


class Config:
    def __init__(self):
        # 获取大模型的密钥
        self.MODEL_API_KEY = os.getenv("MODEL_API_KEY")
        self.MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
        self.MODEL_NAME = os.getenv("MODEL_NAME")
        # 读取neo4j环境变量
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        # 读取潜入模型路径
        self.EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
        # 读取极梦的密钥
        self.JIMENG_AK = os.getenv("JIMENG_AK")
        self.JIMENG_SK = os.getenv("JIMENG_SK")
        # FAISS索引基础路径（解决中文路径在Windows上faiss读取问题）
        self.FAISS_INDEX_BASE = FAISS_INDEX_BASE


if __name__ == '__main__':
    config = Config()
    print(config.MODEL_BASE_URL)
    print(config.MODEL_API_KEY)
