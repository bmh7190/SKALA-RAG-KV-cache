"""시장성 평가 설계서의 정량 채점 규칙.

이 모듈은 웹 검색이나 LLM 판단과 분리한다. 조사 단계가 확인한 수치와
채택/생태계 수준을 넘기면 설계서의 동일한 경계값으로 점수를 계산한다.
"""

from collections.abc import Iterable
from math import isfinite
from typing import Literal, NamedTuple

MARKET_GROWTH = "시장 규모·성장성"
COMMERCIAL_ADOPTION = "상용화·채택 현황"
ECOSYSTEM_SUPPORT = "생태계 지지"
MARKET_CRITERIA = (MARKET_GROWTH, COMMERCIAL_ADOPTION, ECOSYSTEM_SUPPORT)

AdoptionLevel = Literal[
    "multiple_production",
    "single_production",
    "external_integration_or_pilot",
    "public_prototype_only",
    "research_only",
]

SupportKind = Literal[
    "framework_integration",
    "external_implementation",
    "api_or_documentation",
    "active_maintenance",
    "standard_or_industry_support",
]


class EcosystemSupport(NamedTuple):
    """중복 계산을 막기 위한 하나의 외부 생태계 지원 사례."""

    kind: SupportKind
    provider: str


_ADOPTION_SCORES: dict[AdoptionLevel, int] = {
    "multiple_production": 5,
    "single_production": 4,
    "external_integration_or_pilot": 3,
    "public_prototype_only": 2,
    "research_only": 1,
}


def score_market_growth(cagr_percent: float) -> int:
    """관련 시장 CAGR(%)를 설계서의 1~5점 구간으로 변환한다."""
    if not isfinite(cagr_percent):
        raise ValueError("CAGR은 유한한 숫자여야 합니다")
    if cagr_percent >= 30:
        return 5
    if cagr_percent >= 20:
        return 4
    if cagr_percent >= 10:
        return 3
    if cagr_percent > 0:
        return 2
    return 1


def score_commercial_adoption(level: AdoptionLevel) -> int:
    """외부 채택 수준을 설계서의 1~5점으로 변환한다."""
    try:
        return _ADOPTION_SCORES[level]
    except KeyError as error:
        raise ValueError(f"알 수 없는 채택 수준: {level}") from error


def independent_supports(supports: Iterable[EcosystemSupport]) -> set[EcosystemSupport]:
    """같은 제공자의 같은 지원 유형을 하나의 사례로만 센다."""
    normalized: set[EcosystemSupport] = set()
    for support in supports:
        provider = " ".join(support.provider.split()).casefold()
        if not provider:
            raise ValueError("생태계 지원 제공자 이름이 필요합니다")
        normalized.add(EcosystemSupport(support.kind, provider))
    return normalized


def score_ecosystem_support(supports: Iterable[EcosystemSupport]) -> int:
    """독립 지원 사례 수를 설계서의 1~5점으로 변환한다."""
    count = len(independent_supports(supports))
    if count >= 4:
        return 5
    if count == 3:
        return 4
    if count == 2:
        return 3
    if count == 1:
        return 2
    return 1
