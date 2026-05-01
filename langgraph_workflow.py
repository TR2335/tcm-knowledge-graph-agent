from langgraph.graph import StateGraph, END
from my_agents.cypher_query_generator_agent import CypherQueryGeneratorAgent
from my_agents.direct_llm_answer_agent import DirectLLMAnswerAgent
from my_agents.effect_checker_agent import EffectCheckerAgent
from my_agents.effect_entity_embedding_matcher_agent import EffectEntityEmbeddingMatcherAgent
from my_agents.final_answer_mixer_agent import FinalAnswerMixerAgent
from my_agents.symptom_or_disease_checker_agent import SymptomOrDiseaseCheckerAgent
from my_agents.symptom_or_disease_entity_embedding_matcher_agent import (
    SymptomOrDiseaseEntityEmbeddingMatcherAgent)
from my_agents.tcm_question_classifier_agent import TCMQuestionClassifierAgent
from my_agents.xiaohongshu_image_generator_agent import XiaohongshuImageGeneratorAgent
from my_agents.xiaohongshu_intent_classifier_agent import XiaohongshuIntentClassifierAgent
from my_agents.xiaohongshu_tcm_post_output_agent import XiaohongshuTCMPostAgent
from agent_state import AgentState
from common.llm import get_session_history

graph = StateGraph(AgentState)

graph.add_node(XiaohongshuIntentClassifierAgent.__name__, XiaohongshuIntentClassifierAgent())
graph.add_node(XiaohongshuTCMPostAgent.__name__, XiaohongshuTCMPostAgent())
graph.add_node(XiaohongshuImageGeneratorAgent.__name__, XiaohongshuImageGeneratorAgent())
graph.add_node(TCMQuestionClassifierAgent.__name__, TCMQuestionClassifierAgent())
graph.add_node(DirectLLMAnswerAgent.__name__, DirectLLMAnswerAgent())
graph.add_node(SymptomOrDiseaseCheckerAgent.__name__, SymptomOrDiseaseCheckerAgent())
graph.add_node(SymptomOrDiseaseEntityEmbeddingMatcherAgent.__name__, SymptomOrDiseaseEntityEmbeddingMatcherAgent())
graph.add_node(EffectCheckerAgent.__name__, EffectCheckerAgent())
graph.add_node(EffectEntityEmbeddingMatcherAgent.__name__, EffectEntityEmbeddingMatcherAgent())
graph.add_node(CypherQueryGeneratorAgent.__name__, CypherQueryGeneratorAgent())
graph.add_node(FinalAnswerMixerAgent.__name__, FinalAnswerMixerAgent())

graph.set_entry_point(XiaohongshuIntentClassifierAgent.__name__)


def xhs_routing(state_agent: AgentState):
    if state_agent['is_has_xhs_intent']:
        return XiaohongshuTCMPostAgent.__name__
    return TCMQuestionClassifierAgent.__name__


graph.add_conditional_edges(XiaohongshuIntentClassifierAgent.__name__, xhs_routing)
graph.add_edge(XiaohongshuTCMPostAgent.__name__, XiaohongshuImageGeneratorAgent.__name__)


def tcm_routing(state_agent: AgentState):
    if state_agent['is_tcm_question']:
        return SymptomOrDiseaseCheckerAgent.__name__
    return DirectLLMAnswerAgent.__name__


graph.add_conditional_edges(TCMQuestionClassifierAgent.__name__, tcm_routing)


def symptom_routing(state_agent: AgentState):
    if state_agent['is_symptom_or_disease'] and state_agent['symptom_or_disease_entities']:
        return SymptomOrDiseaseEntityEmbeddingMatcherAgent.__name__
    return EffectCheckerAgent.__name__


graph.add_conditional_edges(SymptomOrDiseaseCheckerAgent.__name__, symptom_routing)
graph.add_edge(SymptomOrDiseaseEntityEmbeddingMatcherAgent.__name__, EffectCheckerAgent.__name__)


def effect_routing(state_agent: AgentState):
    if state_agent['is_effect_question'] and state_agent['effect_entities']:
        return EffectEntityEmbeddingMatcherAgent.__name__
    return CypherQueryGeneratorAgent.__name__


graph.add_conditional_edges(EffectCheckerAgent.__name__, effect_routing)
graph.add_edge(EffectEntityEmbeddingMatcherAgent.__name__, CypherQueryGeneratorAgent.__name__)
graph.add_edge(CypherQueryGeneratorAgent.__name__, DirectLLMAnswerAgent.__name__)


def answer_routing(state_agent: AgentState):
    if state_agent['is_tcm_question']:
        return FinalAnswerMixerAgent.__name__
    return END


graph.add_conditional_edges(DirectLLMAnswerAgent.__name__, answer_routing)
graph.add_edge(FinalAnswerMixerAgent.__name__, END)

app = graph.compile()


def call_langgraph_ai(user_id, session_id, user_content):
    response = app.invoke({"input": user_content, "user_id": user_id, "session_id": session_id})
    if not response['is_has_xhs_intent']:
        response_output = response['output']
        history = get_session_history(session_id)
        history.add_user_message(user_content)
        history.add_ai_message(response_output)
        return False, (response_output,)
    title = response['xiaohongshu_tcm_post_title']
    content = response['xiaohongshu_tcm_post_content']
    image_path = response['xiaohongshu_tcm_post_image_path']
    history = get_session_history(session_id)
    history.add_user_message(user_content)
    history.add_ai_message(title + content)
    return True, (title, content, image_path)


if __name__ == '__main__':
    call_langgraph_ai("user_001", "session_001", "咳嗽吃什么药呢？")
