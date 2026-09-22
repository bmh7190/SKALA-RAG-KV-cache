"""근거, 평가, 종합 결과를 연결하는 최소 자료형."""

from typing import Literal, NotRequired, TypedDict

Technology = Literal["KIVI", "InfiniGen"]
BasisStatus = Literal["unverified", "source_checked", "inferred", "public_estimate"]


class SourceRef(TypedDict):
    document: str
    url: str | None
    page: int | None  # URL/비페이지 자료는 None


class ExperimentContext(TypedDict, total=False):
    model: str
    workload: str
    baseline: str


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


class MarketMetric(TypedDict):
    market_name: str
    cagr_percent: float
    period_start_year: int
    period_end_year: int
    source_url: str
    base_market_size: NotRequired[float]
    base_year: NotRequired[int]
    forecast_market_size: NotRequired[float]
    forecast_year: NotRequired[int]
    currency_and_unit: NotRequired[str]


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
    market_metric: NotRequired[MarketMetric]


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
