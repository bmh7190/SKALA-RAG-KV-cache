"""전체 그래프에서 공유하는 값. 노드는 자신이 소유한 키만 갱신한다."""

from typing import TypedDict

from kv_cache_eval.common.schemas import (
    EvidenceGap,
    EvaluationResult,
    ReportDraft,
    ResearchResult,
    Synthesis,
    Technology,
)


class DomainAndCriteria(TypedDict):
    domain: str
    criteria: tuple[str, ...]


class State(TypedDict):
    # 입력
    selected_technologies: tuple[Technology, Technology]
    domain_and_criteria: DomainAndCriteria
    max_research_rounds: int
    # 중간 결과: None=아직 생산되지 않음. 빈 결과/낮은 평가와 구분한다.
    kivi_evidence: ResearchResult | None
    infinigen_evidence: ResearchResult | None
    market_evidence: ResearchResult | None
    maturity_eval: EvaluationResult | None
    market_eval: EvaluationResult | None
    stakeholder_eval: EvaluationResult | None
    domain_eval: EvaluationResult | None
    evidence_gaps: list[EvidenceGap] | None  # None=미검토, []=검토 후 공백 없음
    research_round: int
    synthesis: Synthesis | None
    report: ReportDraft | None


class StateUpdate(TypedDict, total=False):
    kivi_evidence: ResearchResult | None
    infinigen_evidence: ResearchResult | None
    market_evidence: ResearchResult | None
    maturity_eval: EvaluationResult | None
    market_eval: EvaluationResult | None
    stakeholder_eval: EvaluationResult | None
    domain_eval: EvaluationResult | None
    evidence_gaps: list[EvidenceGap] | None
    research_round: int
    synthesis: Synthesis | None
    report: ReportDraft | None


def new_state(max_research_rounds: int = 2) -> State:
    """한 실행의 독립적인 초기값을 만든다. 횟수는 최초 조사 이후 추가 조사 횟수다."""
    if max_research_rounds < 0:
        raise ValueError("max_research_rounds must be >= 0")
    return {
        "selected_technologies": ("KIVI", "InfiniGen"),
        "domain_and_criteria": {
            "domain": "GPU 기반 클라우드 LLM 서비스",
            "criteria": ("기술 성숙도(TRL)", "시장성", "이해관계자", "도메인 적합성"),
        },
        "max_research_rounds": max_research_rounds,
        "kivi_evidence": None,
        "infinigen_evidence": None,
        "market_evidence": None,
        "maturity_eval": None,
        "market_eval": None,
        "stakeholder_eval": None,
        "domain_eval": None,
        "evidence_gaps": None,
        "research_round": 0,
        "synthesis": None,
        "report": None,
    }
