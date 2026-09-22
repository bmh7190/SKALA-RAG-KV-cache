"""입력: 종합과 실제 사용 근거. 출력: report 하나. PDF 작성은 후속 작업."""

import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

from kv_cache_eval.common.state import State, StateUpdate


SYSTEM_PROMPT = """
당신은 KV Cache 최적화 기술에 대한 다관점 평가 보고서를 작성하는
중립적인 기술 보고서 작성자다.

KIVI와 InfiniGen의 기술 근거, 기술 성숙도, 시장성, 이해관계자,
도메인 적용성 및 종합 평가 결과를 바탕으로 보고서를 작성한다.

다음 원칙을 반드시 준수하라.

1. 입력으로 제공된 자료와 근거만 사용한다.
2. KIVI와 InfiniGen의 우열을 판정하거나 하나의 기술을 추천하지 않는다.
3. 서로 다른 실험 조건에서 측정된 수치를 직접 비교하지 않는다.
4. 수치에는 가능한 경우 모델, 데이터셋, 하드웨어, 문맥 길이,
   배치 크기 등 확인된 실험 조건을 함께 작성한다.
5. 직접 확인된 사실과 기술 자료로부터 추론한 영향을 구분한다.
6. 입력에 source_id가 있으면 해당 주장의 끝에 [source_id] 형식으로 표시한다.
7. 입력에 없는 출처, 수치, 기업 사례, 시장 규모 및 도입 사례를 생성하지 않는다.
8. 근거가 부족하거나 확인되지 않은 내용은 한계점에 명시한다.
9. 관점별 평가가 일치하는 부분뿐 아니라 상충하는 부분도 명확히 작성한다.
10. REFERENCE에는 본문에서 실제로 인용한 source_id만 포함한다.
11. SUMMARY는 보고서 소개가 아니라 전체 평가 결과의 핵심 요약으로 작성한다.
12. SUMMARY는 전체 보고서의 1/2페이지를 넘지 않을 정도로 간결하게 작성한다.
13. 보고서는 한국어 Markdown 형식으로 작성한다.

보고서 목차는 반드시 다음 순서를 따른다.

# SUMMARY

# 1. 분석 배경
## 1.1 KV Cache의 역할
## 1.2 KV Cache 병목 문제
## 1.3 분석 목적

# 2. 평가 대상 기술 선정
## 2.1 KIVI
## 2.2 InfiniGen
## 2.3 두 기술의 접근 방식 비교

# 3. 기술 개요
## 3.1 KIVI
## 3.2 InfiniGen
## 3.3 실험 결과 해석 기준

# 4. 관점별 평가
## 4.1 기술 성숙도
## 4.2 시장성
## 4.3 이해관계자
## 4.4 도메인 적용

# 5. 관점 종합 및 시사점
## 5.1 관점별 평가 요약
## 5.2 관점 간 차이 및 Trade-off
## 5.3 적용 조건에 따른 해석

# 6. 분석 한계 및 편향 방지
## 6.1 공개 정보 기반 평가의 한계
## 6.2 실험 조건 차이
## 6.3 확증편향 방지 조치

# REFERENCE
""".strip()


def _to_serializable(value: Any) -> Any:
    """State 값을 JSON 직렬화 가능한 형태로 변환한다."""

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
    """환경변수에 설정된 모델로 보고서 Agent용 LLM을 생성한다."""

    model_name = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or "gpt-4o-mini"
    )

    return ChatOpenAI(
        model=model_name,
        temperature=0,
    )


def _get_response_text(response: Any) -> str:
    """LLM 응답에서 문자열 본문을 추출한다."""

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


def _collect_source_ids(value: Any) -> set[str]:
    """
    입력 자료에서 실제 존재하는 source_id를 재귀적으로 수집한다.

    보고서 Agent가 존재하지 않는 출처 식별자를 생성하지 않도록
    허용 가능한 source_id 목록을 만든다.
    """

    source_ids: set[str] = set()

    if value is None:
        return source_ids

    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()

    if isinstance(value, dict):
        source_id = value.get("source_id")

        if source_id:
            source_ids.add(str(source_id))

        for item in value.values():
            source_ids.update(_collect_source_ids(item))

    elif isinstance(value, (list, tuple, set)):
        for item in value:
            source_ids.update(_collect_source_ids(item))

    return source_ids


def write_report(state: State) -> StateUpdate:
    """
    기술 근거와 관점별 평가 결과를 최종 Markdown 보고서로 작성한다.

    입력 State:
        selected_technologies:
            Human이 선정한 KIVI와 InfiniGen.
        domain_and_criteria:
            평가 도메인 및 관점별 평가 기준.
        kivi_evidence:
            KIVI의 원리, 실험 조건·결과, 한계와 출처.
        infinigen_evidence:
            InfiniGen의 원리, 실험 조건·결과, 한계와 출처.
        maturity_eval:
            기술 성숙도 및 TRL 평가.
        market_eval:
            시장 성장성, 상용화·채택, 생태계 평가.
        stakeholder_eval:
            이해관계자별 이점, 부담과 근거 수준.
        domain_eval:
            GPU 기반 클라우드 LLM 서비스 환경의 적용성 평가.
        evidence_gaps:
            출처가 없거나 확인되지 않은 주장.
        synthesis:
            관점별 공통점, 차이점, 상충 관계와 적용 조건.

    출력 State:
        report:
            SUMMARY부터 REFERENCE까지 포함한 Markdown 보고서.
    """

    report_data = {
        "selected_technologies": _to_serializable(
            state.get("selected_technologies")
        ),
        "domain_and_criteria": _to_serializable(
            state.get("domain_and_criteria")
        ),
        "kivi_evidence": _to_serializable(
            state.get("kivi_evidence")
        ),
        "infinigen_evidence": _to_serializable(
            state.get("infinigen_evidence")
        ),
        "maturity_eval": _to_serializable(
            state.get("maturity_eval")
        ),
        "market_eval": _to_serializable(
            state.get("market_eval")
        ),
        "stakeholder_eval": _to_serializable(
            state.get("stakeholder_eval")
        ),
        "domain_eval": _to_serializable(
            state.get("domain_eval")
        ),
        "evidence_gaps": _to_serializable(
            state.get("evidence_gaps", [])
        ),
        "synthesis": _to_serializable(
            state.get("synthesis")
        ),
    }

    required_results = {
        "maturity_eval": state.get("maturity_eval"),
        "market_eval": state.get("market_eval"),
        "stakeholder_eval": state.get("stakeholder_eval"),
        "domain_eval": state.get("domain_eval"),
        "synthesis": state.get("synthesis"),
    }

    missing_results = [
        key
        for key, value in required_results.items()
        if not value
    ]

    allowed_source_ids = sorted(
        _collect_source_ids(report_data)
    )

    user_prompt = f"""
아래 자료를 이용하여 최종 다관점 평가 보고서를 작성하라.

<보고서 입력 자료>
{json.dumps(report_data, ensure_ascii=False, indent=2)}
</보고서 입력 자료>

<누락된 평가 결과>
{json.dumps(missing_results, ensure_ascii=False)}
</누락된 평가 결과>

<사용 가능한 source_id>
{json.dumps(allowed_source_ids, ensure_ascii=False, indent=2)}
</사용 가능한 source_id>

작성 지침:

1. 누락된 평가 결과가 있으면 내용을 임의로 보완하지 않는다.
2. 누락된 내용은 '6. 분석 한계 및 편향 방지'에 명시한다.
3. 본문의 인용에는 위의 사용 가능한 source_id만 사용할 수 있다.
4. source_id가 없는 근거는 출처가 확인되지 않은 것으로 표시한다.
5. REFERENCE에는 본문에서 실제 사용한 source_id와 입력 자료에
   포함된 서지정보만 기재한다.
6. 서지정보가 부족하면 임의로 채우지 말고 확인 가능한 정보만 기재한다.
7. 관점별 점수가 있다면 점수와 함께 판정 근거를 설명한다.
8. KIVI는 KV Cache의 저장 크기를 줄이는 접근으로 설명한다.
9. InfiniGen은 호스트 메모리의 KV Cache 중 필요한 항목을
   선택적으로 GPU로 가져오는 접근으로 설명한다.
10. 두 기술이 경쟁 관계이면서 조건에 따라 보완적으로 사용될
    가능성이 있음을 근거 범위 안에서 분석한다.
11. 최종 출력에는 작성 안내나 부가 설명 없이 보고서 본문만 포함한다.
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

    report = _get_response_text(response)

    if not report:
        raise ValueError("보고서 Agent가 빈 결과를 반환했습니다.")

    required_headings = [
        "# SUMMARY",
        "# 1. 분석 배경",
        "# 2. 평가 대상 기술 선정",
        "# 3. 기술 개요",
        "# 4. 관점별 평가",
        "# 5. 관점 종합 및 시사점",
        "# 6. 분석 한계 및 편향 방지",
        "# REFERENCE",
    ]

    missing_headings = [
        heading
        for heading in required_headings
        if heading not in report
    ]

    if missing_headings:
        raise ValueError(
            "보고서에서 필수 목차가 누락되었습니다: "
            + ", ".join(missing_headings)
        )

    return {
        "report": report,
    }