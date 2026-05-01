import os
from dotenv import load_dotenv
from tools.path_utils import resolve_from_project_root

load_dotenv(".env")
load_dotenv(resolve_from_project_root(".env"))

FAISS_INDEX_BASE = os.getenv("FAISS_INDEX_BASE", resolve_from_project_root("__004__langgraph_agent"))


class Config:
    def __init__(self):
        self.MODEL_API_KEY = os.getenv("MODEL_API_KEY")
        self.MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
        self.MODEL_NAME = os.getenv("MODEL_NAME")
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
        self.JIMENG_AK = os.getenv("JIMENG_AK")
        self.JIMENG_SK = os.getenv("JIMENG_SK")
        self.FAISS_INDEX_BASE = FAISS_INDEX_BASE
