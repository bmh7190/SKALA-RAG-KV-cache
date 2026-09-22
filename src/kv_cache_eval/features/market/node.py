"""입력: 양쪽 조사 근거. 출력: market_eval 하나."""

from kv_cache_eval.common.state import State, StateUpdate


def evaluate(state: State) -> StateUpdate:
    """TODO: 시장성 판단의 전제·비교 기준·근거 ID를 연결한다."""
    raise NotImplementedError("시장성 평가 노드를 구현해야 합니다")
