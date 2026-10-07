"""전체 그래프에서 공유하는 값. 노드는 자신이 소유한 키만 갱신한다."""

from typing import Literal, TypedDict
from uuid import uuid4

from kv_cache_eval.common.schemas import (
    AgentName, AgentExecution, AgentError, RetryRequest, QualityResult, EvidenceDecision,
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
    domain_evidence: ResearchResult | None
    maturity_eval: EvaluationResult | None
    market_eval: EvaluationResult | None
    stakeholder_eval: EvaluationResult | None
    domain_eval: EvaluationResult | None
    evidence_gaps: list[EvidenceGap] | None  # None=미검토, []=검토 후 공백 없음
    research_round: int
    synthesis: Synthesis | None
    report: ReportDraft | None
    # Supervisor control; one worker runs at a time, so no append reducers are needed.
    trace_id: str
    question: str
    next_agent: AgentName | None
    retry_request: RetryRequest | None
    completed_agents: list[AgentName]
    pending_work: dict[AgentName, RetryRequest]
    last_result: AgentExecution | None
    last_error: AgentError | None
    step_count: int
    max_steps: int
    agent_calls: dict[AgentName, int]
    max_agent_calls: int
    gap_attempts: dict[str, int]
    report_revision: int
    max_report_revisions: int
    quality_result: QualityResult | None
    evidence_decision: EvidenceDecision | None
    pdf_path: str | None
    status: Literal["running", "completed", "incomplete", "failed"]
    termination_reason: str | None


class StateUpdate(TypedDict, total=False):
    kivi_evidence: ResearchResult | None
    infinigen_evidence: ResearchResult | None
    market_evidence: ResearchResult | None
    domain_evidence: ResearchResult | None
    maturity_eval: EvaluationResult | None
    market_eval: EvaluationResult | None
    stakeholder_eval: EvaluationResult | None
    domain_eval: EvaluationResult | None
    evidence_gaps: list[EvidenceGap] | None
    research_round: int
    synthesis: Synthesis | None
    report: ReportDraft | None
    # Supervisor control; one worker runs at a time, so no append reducers are needed.
    trace_id: str
    question: str
    next_agent: AgentName | None
    retry_request: RetryRequest | None
    completed_agents: list[AgentName]
    pending_work: dict[AgentName, RetryRequest]
    last_result: AgentExecution | None
    last_error: AgentError | None
    step_count: int
    max_steps: int
    agent_calls: dict[AgentName, int]
    max_agent_calls: int
    gap_attempts: dict[str, int]
    report_revision: int
    max_report_revisions: int
    quality_result: QualityResult | None
    evidence_decision: EvidenceDecision | None
    pdf_path: str | None
    status: Literal["running", "completed", "incomplete", "failed"]
    termination_reason: str | None


def new_state(max_research_rounds: int = 2, *, question: str = "", trace_id: str | None = None,
              max_steps: int = 30, max_agent_calls: int = 6,
              max_report_revisions: int = 2) -> State:
    """한 실행의 독립적인 초기값을 만든다. 횟수는 최초 조사 이후 추가 조사 횟수다."""
    if max_research_rounds < 0:
        raise ValueError("max_research_rounds must be >= 0")
    if max_steps < 1 or max_agent_calls < 1 or max_report_revisions < 0:
        raise ValueError("작업 상한은 양수, 보고서 수정 상한은 0 이상이어야 합니다")
    return {
        "trace_id": trace_id or str(uuid4()), "question": question,
        "next_agent": None, "retry_request": None, "completed_agents": [],
        "pending_work": {}, "last_result": None, "last_error": None,
        "step_count": 0, "max_steps": max_steps, "agent_calls": {},
        "max_agent_calls": max_agent_calls, "gap_attempts": {},
        "report_revision": 0, "max_report_revisions": max_report_revisions,
        "quality_result": None, "evidence_decision": None, "pdf_path": None,
        "status": "running", "termination_reason": None,
        "selected_technologies": ("KIVI", "InfiniGen"),
        "domain_and_criteria": {
            "domain": "GPU 기반 클라우드 LLM 서비스",
            "criteria": ("기술 성숙도(TRL)", "시장성", "이해관계자", "도메인 적합성"),
        },
        "max_research_rounds": max_research_rounds,
        "kivi_evidence": None,
        "infinigen_evidence": None,
        "market_evidence": None,
        "domain_evidence": None,
        "maturity_eval": None,
        "market_eval": None,
        "stakeholder_eval": None,
        "domain_eval": None,
        "evidence_gaps": None,
        "research_round": 0,
        "synthesis": None,
        "report": None,
    }
