"""입력: 기술과 평가 도메인. 출력: 시장 근거와 market_eval."""

import os
from collections.abc import Mapping
from typing import NamedTuple

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.llm import chat_model
from kv_cache_eval.common.schemas import Technology
from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.common.tasks import (
    criteria,
    merge_evaluation,
    merge_research,
    technologies,
)
from kv_cache_eval.features.market.analysis import (
    Analyse,
    MarketAnalysis,
    UnknownSourceUrl,
    materialize_analysis,
    merge_results,
)
from kv_cache_eval.features.market.research import (
    Search,
    collect_market_sources,
    tavily_search,
)
from kv_cache_eval.features.market.rubric import MARKET_CRITERIA


class MarketRuntimeConfig(NamedTuple):
    provider: str
    model: str


def validate_runtime_config(environment: Mapping[str, str]) -> MarketRuntimeConfig:
    """실제 API 객체를 만들기 전에 필요한 시장성 실행 설정을 확인한다."""
    required = ("TAVILY_API_KEY", "LLM_PROVIDER", "LLM_MODEL")
    missing = [name for name in required if not environment.get(name, "").strip()]
    provider = environment.get("LLM_PROVIDER", "").strip().casefold()
    if provider == "openai" and not environment.get("OPENAI_API_KEY", "").strip():
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(
            f"시장성 평가 실행 설정이 비어 있습니다: {', '.join(missing)}"
        )
    if provider != "openai":
        raise RuntimeError(
            f"현재 설치 구성에서 지원하지 않는 LLM_PROVIDER입니다: {provider}"
        )
    return MarketRuntimeConfig(
        provider=provider, model=environment["LLM_MODEL"].strip()
    )


def _openai_analyst(
    config: MarketRuntimeConfig, selected_criteria=MARKET_CRITERIA
) -> Analyse:
    llm = chat_model()
    structured = llm.with_structured_output(
        MarketAnalysis, method="json_schema", strict=True
    )

    def analyse(technology: Technology, hits):
        sources = "\n\n".join(
            f"URL: {hit.url}\n제목: {hit.title}\n내용: {hit.content}" for hit in hits
        )
        prompt = f"""당신은 GPU 기반 클라우드 LLM 서비스의 시장성 분석가다.
대상 기술은 {technology}다. 아래 검색 자료에 명시된 사실만 사용하라.

시장 규모·성장성은 관련 AI 추론/AI 인프라 시장 CAGR을 사용한다. 전체 시장 성장을
{technology}의 직접 채택 근거로 해석하지 않는다.
상용화·채택은 다수 운영=multiple_production, 1건 운영=single_production,
외부 통합/Pilot=external_integration_or_pilot, 저자 공개 구현·프로토타입만 있음=
public_prototype_only, 논문 단계=research_only 중 하나다. 확인할 수 없으면 null이다.
생태계 사례는 동일 제공자의 동일 지원 유형을 중복 기록하지 않는다. KIVI에서 영감을 받은
파생 구현은 KIVI 자체의 운영 채택과 구분한다. 자료에 없는 URL은 절대 인용하지 않는다.

평가할 항목: {selected_criteria}. 나머지 항목은 null과 빈 출처 목록으로 둔다.

검색 자료:
{sources}
"""
        for attempt in range(2):
            analysis = MarketAnalysis.model_validate(structured.invoke(prompt))
            try:
                materialize_analysis(technology, hits, analysis)
                return analysis
            except UnknownSourceUrl as error:
                if attempt:
                    raise
                prompt += f"\n인용 검증 실패: {error}. 위 검색 자료의 URL만 그대로 사용하여 다시 작성하라."
        raise AssertionError("unreachable")

    return analyse


def build_market_node(search: Search, analyse: Analyse):
    """테스트와 공급자 교체가 가능하도록 의존성을 주입한 노드를 만든다."""

    def node(state: State) -> StateUpdate:
        results = []
        selected = criteria(state, MARKET_CRITERIA)
        for technology in technologies(state):
            hits = collect_market_sources(search, technology, selected)
            analysis = analyse(technology, hits)
            results.append(materialize_analysis(technology, hits, analysis))
        research, evaluation = merge_results(results)
        evaluation["evaluations"] = [
            row for row in evaluation["evaluations"] if row["criterion"] in selected
        ]
        return {
            "market_evidence": merge_research(state.get("market_evidence"), research),
            "market_eval": merge_evaluation(state, "market_eval", evaluation),
        }

    return node


def evaluate(state: State) -> StateUpdate:
    """Tavily 검색과 구조화 LLM 판단으로 두 기술의 시장성을 평가한다."""
    load_environment()
    config = validate_runtime_config(os.environ)
    return build_market_node(
        tavily_search(), _openai_analyst(config, criteria(state, MARKET_CRITERIA))
    )(state)
