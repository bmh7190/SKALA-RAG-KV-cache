"""입력: 네 평가와 evidence_gaps. 출력: synthesis 하나."""

import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.state import State, StateUpdate


SYSTEM_PROMPT = """
당신은 KV Cache 최적화 기술을 평가하는 중립적인 종합 분석가다.

입력으로 KIVI와 InfiniGen에 대한 다음 평가 결과가 제공된다.

- 기술 성숙도
- 시장성
- 이해관계자
- 도메인 적용성
- 근거가 없거나 확인되지 않은 주장

다음 원칙을 반드시 준수하라.

1. KIVI와 InfiniGen의 우열을 판정하거나 하나의 기술을 추천하지 않는다.
2. 입력으로 제공된 평가 결과와 근거만 사용한다.
3. 서로 다른 모델, 하드웨어, 데이터셋, 문맥 길이, 배치 크기에서
   측정된 성능 수치를 동일 조건의 결과처럼 직접 비교하지 않는다.
4. 직접 확인된 사실과 기술 자료로부터 추론한 영향을 구분한다.
5. 출처가 없거나 확인되지 않은 내용은 미해결 근거 공백에 기록한다.
6. 기술 성숙도, 시장성, 이해관계자, 도메인 적용성 사이의 공통점,
   차이점 및 상충 관계를 명시한다.
7. 각 기술이 유리할 수 있는 적용 조건과 한계를 함께 작성한다.
8. 두 기술의 상호 보완 가능성은 확정된 사실이 아닌
   조건부 가능성으로 작성한다.
9. 입력에 source_id가 있는 경우 해당 주장의 끝에
   [source_id] 형식으로 표시한다.
10. 입력에 없는 출처, 수치, 기업 사례와 도입 사례를 생성하지 않는다.

다음 형식으로 작성하라.

## 1. 관점별 공통점
- 여러 평가 관점에서 공통으로 확인된 내용을 작성한다.

## 2. 관점별 차이와 상충 관계
- 기술 성숙도, 시장성, 이해관계자, 도메인 적용성 평가가
  서로 다르게 나타나는 지점을 작성한다.

## 3. 핵심 Trade-off
### KIVI
- 얻을 수 있는 이점과 함께 발생하는 부담을 작성한다.

### InfiniGen
- 얻을 수 있는 이점과 함께 발생하는 부담을 작성한다.

## 4. 기술별 적용 조건과 한계
### KIVI
- 적합할 수 있는 조건
- 적용 시 한계와 확인 사항

### InfiniGen
- 적합할 수 있는 조건
- 적용 시 한계와 확인 사항

## 5. 상호 보완 가능성
- 두 기술을 함께 적용할 가능성과 추가로 검증해야 하는 내용을 작성한다.

## 6. 미해결 근거 공백
- 자료가 없거나 확인되지 않은 주장과 추가 조사가 필요한 내용을 작성한다.

## 7. 종합 의견
- 우열 판정 없이 관점에 따라 평가가 달라지는 이유를 정리한다.
""".strip()


def _to_serializable(value: Any) -> Any:
    """State 값을 JSON 직렬화가 가능한 형태로 변환한다."""

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    if isinstance(value, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _get_llm() -> ChatOpenAI:
    """환경변수에 설정된 OpenAI 생성 LLM을 생성한다."""

    load_environment()

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if provider != "openai":
        raise ValueError(
            "LLM_PROVIDER는 'openai'여야 합니다. "
            f"현재 값: {provider or '미설정'}"
        )

    if not model_name:
        raise ValueError(
            "LLM_MODEL 환경변수가 설정되지 않았습니다."
        )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
        )

    return ChatOpenAI(
        model=model_name,
    )


def _get_response_text(response: Any) -> str:
    """LLM 응답 객체에서 문자열 본문을 추출한다."""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    text_parts.append(str(text))

            else:
                text_parts.append(str(item))

        return "\n".join(text_parts).strip()

    return str(content).strip()


def synthesize(state: State) -> StateUpdate:
    """
    네 가지 관점별 평가 결과를 종합한다.

    입력 State:
        maturity_eval:
            KIVI와 InfiniGen의 기술 성숙도 및 TRL 평가 결과.
        market_eval:
            시장 성장성, 상용화·채택 및 생태계 평가 결과.
        stakeholder_eval:
            운영자, 개발자, 이용자, 경쟁 기술 진영,
            투자·산업 관계자 관점의 평가 결과.
        domain_eval:
            GPU 기반 클라우드 LLM 서비스 환경의 평가 결과.
        evidence_gaps:
            출처가 없거나 확인되지 않은 주장.

    출력 State:
        synthesis:
            관점별 공통점, 차이, 상충 관계, Trade-off,
            적용 조건, 한계와 근거 공백을 포함한 종합 결과.
    """

    maturity_eval = state.get("maturity_eval")
    market_eval = state.get("market_eval")
    stakeholder_eval = state.get("stakeholder_eval")
    domain_eval = state.get("domain_eval")
    evidence_gaps = state.get("evidence_gaps", [])

    missing_inputs: list[str] = []

    if not maturity_eval:
        missing_inputs.append("maturity_eval")

    if not market_eval:
        missing_inputs.append("market_eval")

    if not stakeholder_eval:
        missing_inputs.append("stakeholder_eval")

    if not domain_eval:
        missing_inputs.append("domain_eval")

    input_data = {
        "maturity_eval": _to_serializable(maturity_eval),
        "market_eval": _to_serializable(market_eval),
        "stakeholder_eval": _to_serializable(
            stakeholder_eval
        ),
        "domain_eval": _to_serializable(domain_eval),
        "evidence_gaps": _to_serializable(evidence_gaps),
        "missing_inputs": missing_inputs,
    }

    user_prompt = f"""
아래는 KIVI와 InfiniGen에 대한 관점별 평가 결과다.

누락된 평가 결과가 있으면 해당 내용을 임의로 보완하지 말고
'미해결 근거 공백'에 명시하라.

<관점별 평가 결과>
{json.dumps(input_data, ensure_ascii=False, indent=2)}
</관점별 평가 결과>

제공된 자료만 사용하여 두 기술의 공통점, 차이점, 상충 관계,
Trade-off, 적용 조건, 한계 및 근거 공백을 종합하라.
""".strip()

    llm = _get_llm()

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
    )

    synthesis = _get_response_text(response)

    if not synthesis:
        raise ValueError(
            "종합 Agent가 빈 결과를 반환했습니다."
        )

    return {
        "synthesis": synthesis,
    }