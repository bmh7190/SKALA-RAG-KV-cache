"""기술별 조사 노드. 입력: 기술, 도메인, 이전 공백. 출력: 해당 기술의 근거 키 하나."""

from kv_cache_eval.common.schemas import Technology
from kv_cache_eval.common.state import State, StateUpdate


def research_technology(state: State, technology: Technology) -> StateUpdate:
    """TODO: 선정 원문에서 Agentic RAG로 주장과 출처를 확인한다."""
    # 두 기술에 동일한 조사 흐름을 쓰되, 결과는 서로 다른 State 키에 기록한다.
    # 재조사 시 evidence_gaps에서 해당 기술의 공백만 찾아 질의를 좁힌다.
    raise NotImplementedError(f"{technology} 기술 조사 노드를 구현해야 합니다")


def research_kivi(state: State) -> StateUpdate:
    return research_technology(state, "KIVI")


def research_infinigen(state: State) -> StateUpdate:
    return research_technology(state, "InfiniGen")
