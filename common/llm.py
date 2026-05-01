from typing import List
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from common.config import Config

conf = Config()

my_llm = ChatOpenAI(
    api_key=conf.MODEL_API_KEY,
    base_url=conf.MODEL_BASE_URL,
    model=conf.MODEL_NAME
)

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def get_top_k_history(session_id: str, k: int = 2) -> List[BaseMessage]:
    history = get_session_history(session_id)
    return history.messages[-k:] if history.messages else []
