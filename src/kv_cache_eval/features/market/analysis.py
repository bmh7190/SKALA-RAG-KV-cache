"""검색 근거를 구조화된 시장 평가와 공통 Evidence로 변환한다."""

from collections.abc import Callable, Iterable
from hashlib import sha256

from pydantic import BaseModel, Field

from kv_cache_eval.common.schemas import (
    Evidence,
    Evaluation,
    EvaluationResult,
    ResearchResult,
    Technology,
)
from kv_cache_eval.features.market.research import (
    SearchHit,
    canonical_url,
)
from kv_cache_eval.features.market.rubric import (
    COMMERCIAL_ADOPTION,
    ECOSYSTEM_SUPPORT,
    MARKET_GROWTH,
    AdoptionLevel,
    EcosystemSupport,
    SupportKind,
    score_commercial_adoption,
    score_ecosystem_support,
    score_market_growth,
)


class CitedAssessment(BaseModel):
    """근거 URL을 포함하는 시장 평가 결과."""

    judgment: str | None = None
    rationale: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    uncertainty: str | None = None


class SupportFinding(BaseModel):
    """프레임워크, 도구, 기업 등의 생태계 지원 근거."""

    kind: SupportKind
    provider: str
    source_url: str


class MarketAnalysis(BaseModel):
    """한 기술에 대해 LLM이 반환해야 하는 제한된 구조."""

    cagr_percent: float | None = None
    growth: CitedAssessment
    adoption_level: AdoptionLevel | None = None
    adoption: CitedAssessment
    supports: list[SupportFinding] = Field(default_factory=list)
    ecosystem: CitedAssessment


Analyse = Callable[
    [Technology, list[SearchHit]],
    MarketAnalysis,
]


def _evidence_id(
    technology: Technology,
    url: str,
) -> str:
    """기술명과 URL로 재현 가능한 Evidence ID를 만든다."""
    digest = sha256(
        url.encode("utf-8")
    ).hexdigest()[:12]

    return f"market-{technology.casefold()}-{digest}"


def _checked_urls(
    urls: Iterable[str],
    hits_by_url: dict[str, SearchHit],
) -> list[str]:
    """
    LLM이 인용한 URL 중 실제 검색 결과에 포함된 URL만 반환한다.

    검색 결과에 없는 URL은 예외를 발생시키지 않고 제외한다.
    """
    checked: list[str] = []

    for raw_url in urls:
        url = canonical_url(raw_url)

        if url not in hits_by_url:
            continue

        if url not in checked:
            checked.append(url)

    return checked


def _checked_supports(
    supports: Iterable[SupportFinding],
    hits_by_url: dict[str, SearchHit],
) -> list[tuple[SupportFinding, str]]:
    """
    실제 검색 결과로 확인된 생태계 지원 사례만 반환한다.

    검증되지 않은 URL에 연결된 지원 사례가 생태계 점수에
    반영되지 않도록 SupportFinding 자체를 걸러낸다.
    """
    checked: list[tuple[SupportFinding, str]] = []
    seen: set[tuple[SupportKind, str, str]] = set()

    for support in supports:
        url = canonical_url(support.source_url)

        if url not in hits_by_url:
            continue

        unique_key = (
            support.kind,
            support.provider,
            url,
        )

        if unique_key in seen:
            continue

        seen.add(unique_key)
        checked.append(
            (
                support,
                url,
            )
        )

    return checked


def _evaluation(
    technology: Technology,
    criterion: str,
    assessment: CitedAssessment,
    score: float | None,
    urls: list[str],
    sources_checked: bool,
) -> Evaluation:
    """검증된 URL과 평가 결과를 공통 Evaluation으로 변환한다."""
    has_basis = bool(
        urls
        and sources_checked
        and (
            assessment.judgment is not None
            or score is not None
        )
    )

    return {
        "technology": technology,
        "criterion": criterion,
        "judgment": assessment.judgment,
        "score": score,
        "rationale": assessment.rationale,
        "evidence_ids": [
            _evidence_id(
                technology,
                url,
            )
            for url in urls
        ],
        "uncertainty": assessment.uncertainty,
        "basis_status": (
            "source_checked"
            if has_basis
            else "unverified"
        ),
    }


def materialize_analysis(
    technology: Technology,
    hits: list[SearchHit],
    analysis: MarketAnalysis,
) -> tuple[ResearchResult, EvaluationResult]:
    """
    LLM 출력을 검증하고 공통 State 자료형으로 변환한다.

    검색 결과에 없는 URL은 제외하며, 유효한 근거가 남지 않으면
    해당 평가는 unverified 상태가 된다.
    """
    hits_by_url = {
        canonical_url(hit.url): hit
        for hit in hits
    }

    growth_urls = _checked_urls(
        analysis.growth.source_urls,
        hits_by_url,
    )

    adoption_urls = _checked_urls(
        analysis.adoption.source_urls,
        hits_by_url,
    )

    checked_supports = _checked_supports(
        analysis.supports,
        hits_by_url,
    )

    support_urls = [
        url
        for _, url in checked_supports
    ]

    ecosystem_assessment_urls = _checked_urls(
        analysis.ecosystem.source_urls,
        hits_by_url,
    )

    ecosystem_urls = list(
        dict.fromkeys(
            (
                *ecosystem_assessment_urls,
                *support_urls,
            )
        )
    )

    cited_urls = list(
        dict.fromkeys(
            (
                *growth_urls,
                *adoption_urls,
                *ecosystem_urls,
            )
        )
    )

    evidence: list[Evidence] = []

    for url in cited_urls:
        hit = hits_by_url[url]

        evidence.append(
            {
                "id": _evidence_id(
                    technology,
                    url,
                ),
                "technology": technology,
                "claim": hit.content,
                "excerpt": hit.content,
                "source": {
                    "document": hit.title,
                    "url": url,
                    "page": None,
                },
                "experiment": None,
                "limitations": [
                    "웹 공개 자료 기반이며 비공개 도입 사례는 "
                    "반영되지 않을 수 있음"
                ],
                "verification_status": (
                    "source_checked"
                    if hit.source_checked
                    else "unverified"
                ),
            }
        )

    growth_score = (
        None
        if analysis.cagr_percent is None
        else score_market_growth(
            analysis.cagr_percent
        )
    )

    adoption_score = (
        None
        if analysis.adoption_level is None
        else score_commercial_adoption(
            analysis.adoption_level
        )
    )

    ecosystem_supports = [
        EcosystemSupport(
            support.kind,
            support.provider,
        )
        for support, _ in checked_supports
    ]

    ecosystem_score = (
        score_ecosystem_support(
            ecosystem_supports
        )
        if ecosystem_urls
        else None
    )

    evaluations = [
        _evaluation(
            technology=technology,
            criterion=MARKET_GROWTH,
            assessment=analysis.growth,
            score=growth_score,
            urls=growth_urls,
            sources_checked=all(
                hits_by_url[url].source_checked
                for url in growth_urls
            ),
        ),
        _evaluation(
            technology=technology,
            criterion=COMMERCIAL_ADOPTION,
            assessment=analysis.adoption,
            score=adoption_score,
            urls=adoption_urls,
            sources_checked=all(
                hits_by_url[url].source_checked
                for url in adoption_urls
            ),
        ),
        _evaluation(
            technology=technology,
            criterion=ECOSYSTEM_SUPPORT,
            assessment=analysis.ecosystem,
            score=ecosystem_score,
            urls=ecosystem_urls,
            sources_checked=all(
                hits_by_url[url].source_checked
                for url in ecosystem_urls
            ),
        ),
    ]

    return (
        {
            "evidence": evidence,
            "notes": [
                "시장성 웹 조사에서 실제 검색 결과로 확인된 "
                "출처만 포함함"
            ],
        },
        {
            "evaluations": evaluations,
            "notes": [],
        },
    )


def merge_results(
    results: Iterable[
        tuple[ResearchResult, EvaluationResult]
    ],
) -> tuple[ResearchResult, EvaluationResult]:
    """기술별 시장 조사 및 평가 결과를 하나로 합친다."""
    evidence: list[Evidence] = []
    evaluations: list[Evaluation] = []
    research_notes: list[str] = []
    evaluation_notes: list[str] = []

    for research, evaluation in results:
        evidence.extend(
            research["evidence"]
        )
        evaluations.extend(
            evaluation["evaluations"]
        )
        research_notes.extend(
            research["notes"]
        )
        evaluation_notes.extend(
            evaluation["notes"]
        )

    return (
        {
            "evidence": evidence,
            "notes": research_notes,
        },
        {
            "evaluations": evaluations,
            "notes": evaluation_notes,
        },
    )