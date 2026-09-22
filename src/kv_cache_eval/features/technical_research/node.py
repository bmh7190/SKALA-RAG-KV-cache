"""선정된 KIVI·InfiniGen에 같은 문서/웹 조사 흐름을 적용하는 Graph 진입점."""

from __future__ import annotations

import os

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.schemas import ResearchResult, Technology
from kv_cache_eval.common.state import State, StateUpdate


def make_runtime_llm():
    """설정한 생성 모델만 사용하며 키 값은 출력하지 않는다."""
    load_environment()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if not provider or not model:
        raise RuntimeError("LLM_PROVIDER와 LLM_MODEL을 .env 또는 셸에 설정하세요")
    if provider != "openai":
        raise RuntimeError("현재 구현된 생성 LLM 제공자는 openai뿐입니다")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY가 필요합니다. 값은 출력하지 않습니다")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, max_retries=0, timeout=45)


def research_technology(
    state: State, technology: Technology, *, settings=None,
    top_k: int = 5, max_attempts: int = 2, max_llm_calls: int = 24,
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
    questions = questions_for_state(state, technology)
    key = "kivi_evidence" if technology == "KIVI" else "infinigen_evidence"
    result = research_questions(questions, rag_search, reviewer, target=technology,
                                top_k=top_k, max_attempts=max_attempts,
                                prior=state[key], web_search=web_search)
    result["notes"].append(
        f"{technology} 조사: 질문 {len(questions)}개, LLM 호출 {reviewer.calls}/{max_llm_calls}, "
        f"외부 재조사 라운드 {state['research_round']}"
    )
    return result


def research_kivi(state: State) -> StateUpdate:
    return {"kivi_evidence": research_technology(state, "KIVI")}


def research_infinigen(state: State) -> StateUpdate:
    return {"infinigen_evidence": research_technology(state, "InfiniGen")}
