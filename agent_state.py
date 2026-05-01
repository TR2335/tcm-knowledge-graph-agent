from typing import TypedDict, List


class AgentState(TypedDict):
    input: str
    user_id: str
    session_id: str
    is_has_xhs_intent: bool
    xiaohongshu_tcm_post_title: str
    xiaohongshu_tcm_post_content: str
    xiaohongshu_tcm_post_strategies: List[str]
    xiaohongshu_tcm_post_image_path: str
    is_tcm_question: bool
    llm_direct_answer: str
    is_symptom_or_disease: bool
    symptom_or_disease_entities: List[str]
    symptom_or_disease_entity_match_results: List[dict]
    is_effect_question: bool
    effect_entities: List[str]
    effect_entity_match_results: List[dict]
    cypher: str
    cypher_result: List
    cypher_answer: str
    output: str
