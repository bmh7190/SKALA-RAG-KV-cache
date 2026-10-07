"""전체 그래프에서 공유하는 값. 노드는 자신이 소유한 키만 갱신한다."""

import re
from typing import Literal, TypedDict
from uuid import uuid4

from kv_cache_eval.common.schemas import (
    AgentError,
    AgentExecution,
    AgentName,
    EvaluationResult,
    EvidenceDecision,
    EvidenceGap,
    QualityResult,
    ReportDraft,
    ResearchResult,
    RetryRequest,
    Synthesis,
    Technology,
)


class DomainAndCriteria(TypedDict):
    domain: str
    criteria: tuple[str, ...]


class SupervisorFields(TypedDict, total=False):
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
    decision_reason: str | None
    termination_reason: str | None


class State(SupervisorFields):
    # 입력
    selected_technologies: tuple[Technology, Technology]
    domain_and_criteria: DomainAndCriteria
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
    synthesis: Synthesis | None
    report: ReportDraft | None


class StateUpdate(SupervisorFields, total=False):
    kivi_evidence: ResearchResult | None
    infinigen_evidence: ResearchResult | None
    market_evidence: ResearchResult | None
    domain_evidence: ResearchResult | None
    maturity_eval: EvaluationResult | None
    market_eval: EvaluationResult | None
    stakeholder_eval: EvaluationResult | None
    domain_eval: EvaluationResult | None
    evidence_gaps: list[EvidenceGap] | None
    synthesis: Synthesis | None
    report: ReportDraft | None


def new_state(
    *,
    question: str = "",
    trace_id: str | None = None,
    max_steps: int = 30,
    max_agent_calls: int = 6,
    max_report_revisions: int = 2,
) -> State:
    """완전한 초기 상태를 만든다. 부분 업데이트와 제어 필드 정의를 공유한다."""
    if max_steps < 1 or max_agent_calls < 1 or max_report_revisions < 0:
        raise ValueError("작업 상한은 양수, 보고서 수정 상한은 0 이상이어야 합니다")
    if trace_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", trace_id):
        raise ValueError("trace_id는 영문·숫자·밑줄·하이픈 1~100자여야 합니다")
    return {
        "trace_id": trace_id or str(uuid4()),
        "question": question,
        "next_agent": None,
        "retry_request": None,
        "completed_agents": [],
        "pending_work": {},
        "last_result": None,
        "last_error": None,
        "step_count": 0,
        "max_steps": max_steps,
        "agent_calls": {},
        "max_agent_calls": max_agent_calls,
        "gap_attempts": {},
        "report_revision": 0,
        "max_report_revisions": max_report_revisions,
        "quality_result": None,
        "evidence_decision": None,
        "pdf_path": None,
        "status": "running",
        "termination_reason": None,
        "decision_reason": None,
        "selected_technologies": ("KIVI", "InfiniGen"),
        "domain_and_criteria": {
            "domain": "GPU 기반 클라우드 LLM 서비스",
            "criteria": ("기술 성숙도(TRL)", "시장성", "이해관계자", "도메인 적합성"),
        },
        "kivi_evidence": None,
        "infinigen_evidence": None,
        "market_evidence": None,
        "domain_evidence": None,
        "maturity_eval": None,
        "market_eval": None,
        "stakeholder_eval": None,
        "domain_eval": None,
        "evidence_gaps": None,
        "synthesis": None,
        "report": None,
    }
