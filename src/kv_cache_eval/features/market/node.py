"""입력: 기술과 평가 도메인. 출력: 시장 근거와 market_eval."""

import os
from collections.abc import Mapping
from typing import NamedTuple

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.schemas import Technology
from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.features.market.analysis import Analyse, MarketAnalysis, materialize_analysis, merge_results
from kv_cache_eval.features.market.research import Search, collect_market_sources, tavily_search


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
        raise RuntimeError(f"시장성 평가 실행 설정이 비어 있습니다: {', '.join(missing)}")
    if provider != "openai":
        raise RuntimeError(f"현재 설치 구성에서 지원하지 않는 LLM_PROVIDER입니다: {provider}")
    return MarketRuntimeConfig(provider=provider, model=environment["LLM_MODEL"].strip())


def _openai_analyst(config: MarketRuntimeConfig) -> Analyse:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=config.model, temperature=0, max_retries=0, timeout=45)
    structured = llm.with_structured_output(MarketAnalysis, method="json_schema", strict=True)

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
저자의 공개 GitHub 구현을 본문에서 확인했다면 외부 운영이 미확인이어도
public_prototype_only로 분류한다. 운영 미확인을 운영 부재로 단정하지 않는다.
생태계 사례는 동일 제공자의 동일 지원 유형을 중복 기록하지 않는다. KIVI에서 영감을 받은
파생 구현은 KIVI 자체의 운영 채택과 구분한다. 자료에 없는 URL은 절대 인용하지 않는다.
supports에는 대상 기술을 직접 지원하는 독립적인 외부 제공자만 넣는다. 저자 논문·저자 코드,
다른 KV 기술의 지원은 제외한다. 외부 소개글 한 건의 여러 기능을 독립 지원 여러 건으로
세지 않는다. 가이드가 vLLM 내부 지원이 아니라고 명시하면 framework_integration이 아니다.
외부 게시글만으로 별도 구현이나 실제 운영을 추정하지 않는다.

검색 자료:
{sources}
"""
        return MarketAnalysis.model_validate(structured.invoke(prompt))

    return analyse


def build_market_node(search: Search, analyse: Analyse):
    """테스트와 공급자 교체가 가능하도록 의존성을 주입한 노드를 만든다."""
    def node(state: State) -> StateUpdate:
        results = []
        for technology in state["selected_technologies"]:
            hits = collect_market_sources(search, technology)
            analysis = analyse(technology, hits)
            results.append(materialize_analysis(technology, hits, analysis))
        research, evaluation = merge_results(results)
        return {"market_evidence": research, "market_eval": evaluation}

    return node


def evaluate(state: State) -> StateUpdate:
    """Tavily 검색과 구조화 LLM 판단으로 두 기술의 시장성을 평가한다."""
    load_environment()
    config = validate_runtime_config(os.environ)
    return build_market_node(tavily_search(), _openai_analyst(config))(state)
