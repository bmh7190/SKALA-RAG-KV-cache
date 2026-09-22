"""입력: 네 평가와 evidence_gaps. 출력: synthesis 하나."""

from kv_cache_eval.common.state import State, StateUpdate


def synthesize(state: State) -> StateUpdate:
    """TODO: 관점별 차이·상충 관계·적용 조건·미해결 공백을 종합한다."""
    raise NotImplementedError("종합 노드를 구현해야 합니다")
