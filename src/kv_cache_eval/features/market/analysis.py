"""검색 근거를 구조화된 시장 평가와 공통 Evidence로 변환한다."""

import re
from collections.abc import Callable, Iterable
from hashlib import sha256
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from kv_cache_eval.common.schemas import (
    Evaluation,
    EvaluationResult,
    Evidence,
    MarketMetric,
    ResearchResult,
    Technology,
)
from kv_cache_eval.features.market.research import SearchHit, canonical_url
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


class SourceCitation(BaseModel):
    source_url: str
    excerpt: str = Field(min_length=8)


class CitedAssessment(BaseModel):
    judgment: str | None = None
    rationale: str | None = None
    citations: list[SourceCitation] = Field(default_factory=list)
    uncertainty: str | None = None


class GrowthMetric(BaseModel):
    market_name: str = Field(min_length=1)
    cagr_percent: float
    period_start_year: int
    period_end_year: int
    citation: SourceCitation
    base_market_size: float | None = None
    base_year: int | None = None
    forecast_market_size: float | None = None
    forecast_year: int | None = None
    currency_and_unit: str | None = None


class SupportFinding(BaseModel):
    kind: SupportKind
    provider: str
    citation: SourceCitation


class MarketAnalysis(BaseModel):
    """한 기술에 대해 LLM이 반환해야 하는 제한된 구조."""

    growth_metric: GrowthMetric | None = None
    growth: CitedAssessment
    adoption_level: AdoptionLevel | None = None
    adoption: CitedAssessment
    supports: list[SupportFinding] = Field(default_factory=list)
    ecosystem: CitedAssessment


Analyse = Callable[[Technology, list[SearchHit]], MarketAnalysis]

_AUTHOR_SOURCE_PREFIXES: dict[Technology, tuple[str, ...]] = {
    "KIVI": (
        "https://github.com/jy-yuan/kivi",
        "https://github.com/strategist922/kivi",
    ),
    "InfiniGen": ("https://github.com/snu-comparch/infinigen",),
}
_PAPER_HOSTS = {
    "arxiv.org",
    "openreview.net",
    "usenix.org",
    "usenix.net",
    "www.usenix.org",
    "www.usenix.net",
}


def _evidence_id(technology: Technology, url: str) -> str:
    digest = sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"market-{technology.casefold()}-{digest}"


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _checked_citations(
    citations: Iterable[SourceCitation],
    hits_by_url: dict[str, SearchHit],
) -> list[tuple[str, str]]:
    checked: list[tuple[str, str]] = []
    for citation in citations:
        url = canonical_url(citation.source_url)
        if url not in hits_by_url:
            raise ValueError(
                f"검색 결과에 없는 URL을 인용했습니다: {citation.source_url}"
            )
        excerpt = " ".join(citation.excerpt.split())
        if _normalized_text(excerpt) not in _normalized_text(hits_by_url[url].content):
            raise ValueError(
                f"출처 본문에서 인용문을 확인할 수 없습니다: {citation.source_url}"
            )
        item = (url, excerpt)
        if item not in checked:
            checked.append(item)
    return checked


def _unique_urls(citations: Iterable[tuple[str, str]]) -> list[str]:
    return list(dict.fromkeys(url for url, _ in citations))


def _is_external_support(technology: Technology, finding: SupportFinding) -> bool:
    url = canonical_url(finding.citation.source_url)
    parts = urlsplit(url)
    source_location = f"{parts.scheme}://{parts.netloc}{parts.path}".casefold().rstrip(
        "/"
    )
    if parts.netloc.casefold() in _PAPER_HOSTS:
        return False
    return not any(
        source_location == prefix or source_location.startswith(f"{prefix}/")
        for prefix in _AUTHOR_SOURCE_PREFIXES[technology]
    )


def _market_metric(
    metric: GrowthMetric | None,
    hits_by_url: dict[str, SearchHit],
) -> MarketMetric | None:
    if metric is None:
        return None
    market_name = metric.market_name.strip()
    if not market_name:
        raise ValueError("CAGR을 적용한 시장 이름이 필요합니다")
    if metric.period_end_year <= metric.period_start_year:
        raise ValueError("CAGR 전망 종료 연도는 시작 연도보다 뒤여야 합니다")
    source_url = canonical_url(metric.citation.source_url)
    source_content = hits_by_url[source_url].content
    numeric_values = {
        "CAGR": f"{metric.cagr_percent:g}",
        "전망 시작 연도": str(metric.period_start_year),
        "전망 종료 연도": str(metric.period_end_year),
    }
    for label, value in numeric_values.items():
        if (
            re.search(
                rf"(?<![\d.]){re.escape(value)}(?!\d)",
                source_content,
            )
            is None
        ):
            raise ValueError(
                f"시장 지표 원문에서 {label} 값을 확인할 수 없습니다: {value}"
            )
    result: MarketMetric = {
        "market_name": market_name,
        "cagr_percent": metric.cagr_percent,
        "period_start_year": metric.period_start_year,
        "period_end_year": metric.period_end_year,
        "source_url": source_url,
    }
    for key in (
        "base_market_size",
        "base_year",
        "forecast_market_size",
        "forecast_year",
        "currency_and_unit",
    ):
        value = getattr(metric, key)
        if value is not None:
            result[key] = value
    return result


def _evaluation(
    technology: Technology,
    criterion: str,
    assessment: CitedAssessment,
    score: float | None,
    urls: list[str],
    sources_checked: bool,
    market_metric: MarketMetric | None = None,
) -> Evaluation:
    has_basis = bool(
        urls
        and sources_checked
        and (assessment.judgment is not None or score is not None)
    )
    result: Evaluation = {
        "technology": technology,
        "criterion": criterion,
        "judgment": assessment.judgment,
        "score": score,
        "rationale": assessment.rationale,
        "evidence_ids": [_evidence_id(technology, url) for url in urls],
        "uncertainty": assessment.uncertainty,
        "basis_status": "source_checked" if has_basis else "unverified",
    }
    if market_metric is not None:
        result["market_metric"] = market_metric
    return result


def materialize_analysis(
    technology: Technology,
    hits: list[SearchHit],
    analysis: MarketAnalysis,
) -> tuple[ResearchResult, EvaluationResult]:
    """LLM 출력을 검증하고 공통 State 자료형으로 변환한다."""
    hits_by_url = {canonical_url(hit.url): hit for hit in hits}
    metric_citations = (
        [] if analysis.growth_metric is None else [analysis.growth_metric.citation]
    )
    growth_citations = _checked_citations(
        [*analysis.growth.citations, *metric_citations],
        hits_by_url,
    )
    adoption_citations = _checked_citations(analysis.adoption.citations, hits_by_url)
    support_citations = _checked_citations(
        (item.citation for item in analysis.supports),
        hits_by_url,
    )
    ecosystem_citations = [
        *_checked_citations(analysis.ecosystem.citations, hits_by_url),
        *support_citations,
    ]

    growth_urls = _unique_urls(growth_citations)
    adoption_urls = _unique_urls(adoption_citations)
    ecosystem_urls = _unique_urls(ecosystem_citations)
    all_citations = [*growth_citations, *adoption_citations, *ecosystem_citations]
    excerpts_by_url: dict[str, list[str]] = {}
    for url, excerpt in all_citations:
        excerpts_by_url.setdefault(url, [])
        if excerpt not in excerpts_by_url[url]:
            excerpts_by_url[url].append(excerpt)

    evidence: list[Evidence] = []
    for url, excerpts in excerpts_by_url.items():
        hit = hits_by_url[url]
        cited_text = " / ".join(excerpts)
        evidence.append(
            {
                "id": _evidence_id(technology, url),
                "technology": technology,
                "claim": cited_text,
                "excerpt": cited_text,
                "source": {"document": hit.title, "url": url, "page": None},
                "experiment": None,
                "limitations": [
                    "웹 공개 자료 기반이며 비공개 도입 사례는 반영되지 않을 수 있음"
                ],
                "verification_status": "source_checked"
                if hit.source_checked
                else "unverified",
            }
        )

    metric = _market_metric(analysis.growth_metric, hits_by_url)
    growth_score = (
        None if metric is None else score_market_growth(metric["cagr_percent"])
    )
    adoption_score = (
        None
        if analysis.adoption_level is None
        else score_commercial_adoption(analysis.adoption_level)
    )
    eligible_supports = [
        item for item in analysis.supports if _is_external_support(technology, item)
    ]
    ecosystem_supports = [
        EcosystemSupport(item.kind, item.provider) for item in eligible_supports
    ]
    ecosystem_score = (
        score_ecosystem_support(ecosystem_supports) if ecosystem_urls else None
    )

    evaluations = [
        _evaluation(
            technology,
            MARKET_GROWTH,
            analysis.growth,
            growth_score,
            growth_urls,
            bool(growth_urls)
            and all(hits_by_url[url].source_checked for url in growth_urls),
            metric,
        ),
        _evaluation(
            technology,
            COMMERCIAL_ADOPTION,
            analysis.adoption,
            adoption_score,
            adoption_urls,
            bool(adoption_urls)
            and all(hits_by_url[url].source_checked for url in adoption_urls),
        ),
        _evaluation(
            technology,
            ECOSYSTEM_SUPPORT,
            analysis.ecosystem,
            ecosystem_score,
            ecosystem_urls,
            bool(ecosystem_urls)
            and all(hits_by_url[url].source_checked for url in ecosystem_urls),
        ),
    ]
    excluded_count = len(analysis.supports) - len(eligible_supports)
    notes = ["시장성 웹 조사에서 실제 평가에 인용한 출처만 포함함"]
    if excluded_count:
        notes.append(
            f"저자 저장소 또는 논문 기반 생태계 지원 {excluded_count}건은 점수에서 제외함"
        )
    return (
        {"evidence": evidence, "notes": notes},
        {"evaluations": evaluations, "notes": []},
    )


def merge_results(
    results: Iterable[tuple[ResearchResult, EvaluationResult]],
) -> tuple[ResearchResult, EvaluationResult]:
    evidence: list[Evidence] = []
    evaluations: list[Evaluation] = []
    research_notes: list[str] = []
    evaluation_notes: list[str] = []
    for research, evaluation in results:
        evidence.extend(research["evidence"])
        evaluations.extend(evaluation["evaluations"])
        research_notes.extend(research["notes"])
        evaluation_notes.extend(evaluation["notes"])
    return (
        {"evidence": evidence, "notes": research_notes},
        {"evaluations": evaluations, "notes": evaluation_notes},
    )
