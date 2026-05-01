from langchain_core.runnables import Runnable
import numpy as np
import faiss
from pathlib import Path
from __004__langgraph_agent.agent_state import AgentState
from common.embedding_model import embedding_model
from common.config import Config

conf = Config()


class EffectEntityEmbeddingMatcherAgent(Runnable):
    def invoke(self, agent_state: AgentState, config: dict = None) -> AgentState:
        entities = agent_state.get('effect_entities', [])
        if not entities:
            agent_state['effect_entity_match_results'] = []
            return agent_state
        index_path = Path(conf.FAISS_INDEX_BASE) / "effect_index.bin"
        meta_path = Path(conf.FAISS_INDEX_BASE) / "effect_meta.json"
        if not index_path.exists() or not meta_path.exists():
            agent_state['effect_entity_match_results'] = []
            return agent_state
        index = faiss.read_index(str(index_path))
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        names = metadata["names"]
        query_vecs = embedding_model.encode(entities, normalize_embeddings=True)
        if len(query_vecs.shape) == 1:
            query_vecs = query_vecs.reshape(1, -1)
        _, indices = index.search(query_vecs, k=3)
        results = []
        for i, entity in enumerate(entities):
            matched = []
            for idx in indices[i]:
                if idx < len(names):
                    matched.append({"matched_entity": names[idx], "score": float(_[i][idx])})
            results.append({"entity": entity, "matches": matched})
        agent_state['effect_entity_match_results'] = results
        return agent_state
