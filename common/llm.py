from typing import List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from common.config import Config

# config设置
conf = Config()

# ============ 配置llm区域 ============
my_llm = ChatOpenAI(
    api_key=conf.MODEL_API_KEY,
    base_url=conf.MODEL_BASE_URL,
    model=conf.MODEL_NAME
)

# 1) 你的“会话历史存储”。生产中建议用 Redis/Mongo/SQL，示例先用内存 dict
store = {}


def get_session_history(session_id: str):
    """获取某个会话的聊天历史"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 获取某个会话的最近 k 条消息
def get_top_k_history(session_id: str, k: int = 2) -> List[BaseMessage]:
    """获取某个会话的最近 k 条消息"""
    history = get_session_history(session_id)  # 你之前写的函数
    return history.messages[-k:] if history.messages else []



