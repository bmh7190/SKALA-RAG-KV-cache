"""그래프 진입 시 비교 대상 기술과 평가 도메인을 검증한다."""


def validate_input(state):
    if tuple(state["selected_technologies"]) != ("KIVI", "InfiniGen"):
        raise ValueError("현재 그래프는 KIVI와 InfiniGen 비교용입니다")
    if not state["domain_and_criteria"]["domain"].strip():
        raise ValueError("평가 도메인이 필요합니다")
    return {}
