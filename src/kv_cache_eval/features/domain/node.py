"""입력: 양쪽 조사 근거와 GPU 클라우드 LLM 도메인. 출력: domain_eval 하나."""

from kv_cache_eval.common.state import State, StateUpdate


def evaluate(state: State) -> StateUpdate:
    """TODO: 적용 조건, 운영 제약, 비교 기준을 근거와 연결한다."""
    raise NotImplementedError("도메인 평가 노드를 구현해야 합니다")
