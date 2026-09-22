"""공통 State를 읽고 infinigen_evidence 하나만 반환하는 어댑터."""

from __future__ import annotations

import os

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.features.technical_research.infinigen.prompts import questions_for_state
from kv_cache_eval.features.technical_research.infinigen.retriever import RetrievalSettings, ensure_index
from kv_cache_eval.features.technical_research.infinigen.workflow import ModelReviewer, research_questions


def make_runtime_llm():
    """제공자·모델·키를 확인한 뒤에만 클라이언트를 만든다. 기본 생성 모델은 없다."""
    load_environment()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if not provider or not model:
        raise RuntimeError("LLM_PROVIDER와 LLM_MODEL을 .env 또는 셸에 설정한 뒤 다시 실행하세요")
    if provider != "openai":
        raise RuntimeError("현재 설치된 생성 LLM 연동은 openai뿐입니다. LLM_PROVIDER=openai를 사용하거나 연동을 추가하세요")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OpenAI 조사를 실행하려면 OPENAI_API_KEY가 필요합니다. 값은 출력하지 않습니다")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, max_retries=0, timeout=45)


def research_infinigen(
    state: State, *, settings: RetrievalSettings | None = None,
    top_k: int = 5, max_attempts: int = 2, max_llm_calls: int = 18,
) -> StateUpdate:
    llm = make_runtime_llm()
    retriever, _ = ensure_index(settings=settings)
    reviewer = ModelReviewer(llm, max_calls=max_llm_calls)
    questions = questions_for_state(state)
    result = research_questions(questions, retriever, reviewer, top_k=top_k,
                                max_attempts=max_attempts, prior=state["infinigen_evidence"])
    result["notes"].append(f"InfiniGen 조사: 질문 {len(questions)}개, LLM 호출 {reviewer.calls}/{max_llm_calls}, 외부 재조사 라운드 {state['research_round']}")
    return {"infinigen_evidence": result}
