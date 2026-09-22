"""입력: 종합과 실제 사용 근거. 출력: report 하나. PDF 작성은 후속 작업."""

from kv_cache_eval.common.state import State, StateUpdate


def write_report(state: State) -> StateUpdate:
    """TODO: SUMMARY부터 REFERENCE까지 작성하고 실제 사용 ID만 인용한다."""
    raise NotImplementedError("보고서 노드를 구현해야 합니다")
