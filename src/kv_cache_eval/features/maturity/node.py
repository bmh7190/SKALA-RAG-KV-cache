"""입력: 양쪽 조사 근거. 출력: maturity_eval 하나."""

from kv_cache_eval.common.state import State, StateUpdate


def evaluate(state: State) -> StateUpdate:
    """TODO: 공개 정보로 TRL을 추정하고 근거 ID·불확실성을 남긴다.

    미공개 정보가 있다는 이유만으로 낮은 TRL을 부여하지 않는다.
    """
    raise NotImplementedError("TRL 평가 노드를 구현해야 합니다")
