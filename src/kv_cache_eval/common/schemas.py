"""근거, 평가, 종합 결과를 연결하는 최소 자료형."""

from typing import Literal

from typing_extensions import NotRequired, TypedDict

Technology = Literal["KIVI", "InfiniGen"]
BasisStatus = Literal["unverified", "source_checked", "inferred", "public_estimate"]


class SourceRef(TypedDict):
    document: str
    url: str | None
    page: int | None  # URL/비페이지 자료는 None


class ExperimentContext(TypedDict, total=False):
    model: str | None
    workload: str | None
    baseline: str | None


class Evidence(TypedDict):
    id: str
    technology: Technology
    claim: str
    excerpt: str | None
    source: SourceRef
    experiment: ExperimentContext | None
    limitations: list[str]
    verification_status: Literal["unverified", "source_checked"]


class ResearchResult(TypedDict):
    evidence: list[Evidence]
    notes: list[str]


class Evaluation(TypedDict):
    technology: Technology
    criterion: str
    judgment: str | None
    score: float | None  # None=미판단. 낮은 점수(0 포함)와 구별한다.
    rationale: str | None
    evidence_ids: list[str]
    uncertainty: str | None
    basis_status: BasisStatus
    stakeholder_group: NotRequired[str]


class EvaluationResult(TypedDict):
    evaluations: list[Evaluation]
    notes: list[str]
    text: NotRequired[str]  # 평가 전체를 설명하는 글. 기존 평가 노드는 생략 가능.


class EvidenceGap(TypedDict):
    technology: Technology | None
    criterion: str
    reason: str


class Synthesis(TypedDict):
    perspective_differences: list[str]
    tradeoffs: list[str]
    application_conditions: list[str]
    unresolved_gaps: list[EvidenceGap]
    cited_evidence_ids: list[str]


class ReportDraft(TypedDict):
    sections: list[tuple[str, str]]  # SUMMARY로 시작, REFERENCE로 끝나야 한다.
    cited_evidence_ids: list[str]


AgentName = Literal[
    "technical_research",
    "maturity",
    "market",
    "stakeholders",
    "domain",
    "synthesis",
    "report",
    "quality",
    "export_pdf",
]


class RetryRequest(TypedDict):
    technology: Technology | None
    criteria: list[str]
    reason: str


class AgentError(TypedDict):
    code: str
    message: str
    retryable: bool


class AgentExecution(TypedDict):
    agent: AgentName
    step: int
    status: Literal["completed", "needs_evidence", "failed"]
    gaps: list[EvidenceGap]
    error: AgentError | None
    changed_keys: NotRequired[list[str]]


QualityCriterion = Literal[
    "groundedness", "neutrality", "bias_control", "perspective_coverage"
]
CheckStatus = Literal["pass", "fail", "unknown"]


class QualityIssue(TypedDict):
    criterion: QualityCriterion
    section: str
    reason: str
    required_action: str


class QualityResult(TypedDict):
    report_revision: int
    passed: bool
    checks: dict[QualityCriterion, CheckStatus]
    reasons: dict[QualityCriterion, str]
    issues: list[QualityIssue]


class EvidenceDecision(TypedDict):
    ready: bool
    reason: str
    blocking_gaps: list[EvidenceGap]
