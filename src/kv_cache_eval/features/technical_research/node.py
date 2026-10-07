"""선정된 KIVI·InfiniGen에 같은 문서/웹 조사 흐름을 적용하는 Graph 진입점."""

from __future__ import annotations


from kv_cache_eval.common.tasks import technologies
from kv_cache_eval.common.schemas import ResearchResult, Technology
from kv_cache_eval.common.state import State, StateUpdate


def make_runtime_llm():
    from kv_cache_eval.common.llm import chat_model
    return chat_model(timeout=90)


def research_technology(
    state: State, technology: Technology, *, settings=None,
    top_k: int = 5, max_attempts: int = 2, max_llm_calls: int = 24,
    user_question: str | None = None,
) -> ResearchResult:
    """기술 이름과 문서 manifest만 바꿔 공통 LangGraph 조사를 실행한다."""
    from kv_cache_eval.features.technical_research.ingest import TECHNOLOGIES
    from kv_cache_eval.features.technical_research.prompts import questions_for_state
    from kv_cache_eval.features.technical_research.retriever import ensure_index
    from kv_cache_eval.features.technical_research.web import search_web
    from kv_cache_eval.features.technical_research.workflow import ModelReviewer, research_questions

    if technology not in TECHNOLOGIES:
        raise ValueError(f"지원하지 않는 조사 기술: {technology}")
    llm = make_runtime_llm()
    retriever = None

    def rag_search(query: str, count: int):
        nonlocal retriever
        if retriever is None:
            retriever, _ = ensure_index(technology, settings=settings)
        return retriever.search(query, count)

    def web_search(query: str, count: int):
        return search_web(query, technology, count)

    reviewer = ModelReviewer(llm, max_calls=max_llm_calls, target=technology)
    questions = questions_for_state(state, technology, user_question=user_question)
    key = "kivi_evidence" if technology == "KIVI" else "infinigen_evidence"
    result = research_questions(questions, rag_search, reviewer, target=technology,
                                top_k=top_k, max_attempts=max_attempts,
                                prior=state[key], web_search=web_search)
    result["notes"].append(
        f"{technology} 조사: 질문 {len(questions)}개, LLM 호출 {reviewer.calls}/{max_llm_calls}, "
        f"외부 재조사 라운드 {state['research_round']}"
    )
    return result


def research(state: State, *, user_question: str | None = None) -> StateUpdate:
    """한 Graph 노드에서 선정 기술 모두를 같은 조사 엔진으로 처리한다."""
    updates: StateUpdate = {}
    for technology in technologies(state):
        key = "kivi_evidence" if technology == "KIVI" else "infinigen_evidence"
        updates[key] = research_technology(state, technology, user_question=user_question)
    return updates
