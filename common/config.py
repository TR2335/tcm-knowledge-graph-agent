import os
from dotenv import load_dotenv
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

load_dotenv(_project_root / ".env")

FAISS_INDEX_BASE = os.getenv("FAISS_INDEX_BASE", str(_project_root))


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
