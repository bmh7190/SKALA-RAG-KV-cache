"""입력: 양쪽 조사 근거. 출력: stakeholder_eval 하나."""

from kv_cache_eval.common.state import State, StateUpdate


def evaluate(state: State) -> StateUpdate:
    """TODO: 직접 확인된 반응과 근거 기반 추론·미확인을 구별한다."""
    raise NotImplementedError("이해관계자 평가 노드를 구현해야 합니다")
