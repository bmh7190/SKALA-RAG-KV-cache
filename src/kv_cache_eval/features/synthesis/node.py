"""입력: 네 평가와 evidence_gaps. 출력: synthesis 하나."""

import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.schemas import EvidenceGap, Synthesis
from kv_cache_eval.common.state import State, StateUpdate


def _get_llm() -> ChatOpenAI:
    """환경변수에 지정된 OpenAI 생성 모델을 만든다."""
    load_environment()

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if provider != "openai":
        raise ValueError(
            "synthesis 노드는 LLM_PROVIDER=openai 설정이 필요합니다."
        )

    if not model_name:
        raise ValueError("LLM_MODEL 환경변수가 설정되지 않았습니다.")

    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
    )


def _require_evaluation(
    state: State,
    key: str,
) -> dict[str, Any]:
    """필수 평가 결과가 존재하는지 확인한다."""
    value = state.get(key)

    if value is None:
        raise ValueError(
            f"{key}가 없습니다. 모든 관점별 평가가 완료된 후 "
            "synthesis 노드를 실행해야 합니다."
        )

    return value


def _collect_valid_evidence_ids(state: State) -> set[str]:
    """기술 조사와 시장성 조사 결과에 실제 존재하는 근거 ID를 수집한다."""
    evidence_ids: set[str] = set()

    for key in (
        "kivi_evidence",
        "infinigen_evidence",
        "market_evidence",
    ):
        research_result = state.get(key)

        if research_result is None:
            continue

        for evidence in research_result.get("evidence", []):
            evidence_id = evidence.get("id")

            if evidence_id:
                evidence_ids.add(evidence_id)

    return evidence_ids


def _normalize_gaps(state: State) -> list[EvidenceGap]:
    """검토되지 않은 공백과 검토 후 공백 없음 상태를 구별한다."""
    evidence_gaps = state.get("evidence_gaps")

    if evidence_gaps is None:
        return [
            {
                "technology": None,
                "criterion": "전체 근거 검토",
                "reason": "evidence_gaps가 아직 검토되지 않았습니다.",
            }
        ]

    return list(evidence_gaps)


def _validate_synthesis(
    synthesis: Synthesis,
    valid_evidence_ids: set[str],
    evidence_gaps: list[EvidenceGap],
) -> Synthesis:
    """LLM 결과를 스키마에 맞게 정규화하고 잘못된 근거 ID를 제거한다."""
    cited_ids = [
        evidence_id
        for evidence_id in synthesis.get("cited_evidence_ids", [])
        if evidence_id in valid_evidence_ids
    ]

    return {
        "perspective_differences": [
            str(item)
            for item in synthesis.get("perspective_differences", [])
            if str(item).strip()
        ],
        "tradeoffs": [
            str(item)
            for item in synthesis.get("tradeoffs", [])
            if str(item).strip()
        ],
        "application_conditions": [
            str(item)
            for item in synthesis.get("application_conditions", [])
            if str(item).strip()
        ],
        # 미해결 공백은 LLM이 임의로 만들지 않고
        # evidence_check 노드가 확정한 값을 그대로 사용한다.
        "unresolved_gaps": evidence_gaps,
        "cited_evidence_ids": list(dict.fromkeys(cited_ids)),
    }


def synthesize(state: State) -> StateUpdate:
    """관점별 평가의 차이·상충 관계·적용 조건을 중립적으로 종합한다."""
    maturity_eval = _require_evaluation(state, "maturity_eval")
    market_eval = _require_evaluation(state, "market_eval")
    stakeholder_eval = _require_evaluation(state, "stakeholder_eval")
    domain_eval = _require_evaluation(state, "domain_eval")

    evidence_gaps = _normalize_gaps(state)
    valid_evidence_ids = _collect_valid_evidence_ids(state)

    synthesis_input = {
        "selected_technologies": state["selected_technologies"],
        "domain_and_criteria": state["domain_and_criteria"],
        "maturity_eval": maturity_eval,
        "market_eval": market_eval,
        "stakeholder_eval": stakeholder_eval,
        "domain_eval": domain_eval,
        "evidence_gaps": evidence_gaps,
        "valid_evidence_ids": sorted(valid_evidence_ids),
    }

    prompt = f"""
당신은 KV cache 최적화 기술 비교평가의 종합 분석 담당자입니다.

비교 대상:
- 소프트웨어 접근: KIVI
- 메모리 오프로딩 접근: InfiniGen

분석 도메인:
- GPU 기반 클라우드 LLM 서비스

아래 관점별 평가 결과를 종합하십시오.

반드시 지킬 원칙:
1. 두 기술의 우열이나 최종 승자를 결정하지 마십시오.
2. 기술 성숙도, 시장성, 이해관계자, 도메인 적용성에 따라
   평가가 어떻게 달라지는지 설명하십시오.
3. 공통점, 차이점, 상충 관계를 명확히 구분하십시오.
4. KIVI와 InfiniGen의 적용 조건과 한계를 함께 정리하십시오.
5. 평가 결과의 basis_status와 uncertainty를 고려하십시오.
6. source_checked, inferred, public_estimate, unverified를 구분하십시오.
7. valid_evidence_ids에 포함되지 않은 근거 ID를 만들지 마십시오.
8. 공개되지 않은 정보는 사실처럼 단정하지 마십시오.
9. KIVI와 InfiniGen을 함께 사용하는 보완 가능성도 검토하되,
   제공된 근거로 판단할 수 없는 내용은 단정하지 마십시오.
10. unresolved_gaps는 입력된 evidence_gaps를 기반으로 판단하십시오.
11. 모든 결과는 한국어로 작성하십시오.

입력 데이터:
{json.dumps(synthesis_input, ensure_ascii=False, indent=2)}

다음 항목을 포함한 구조화 결과를 생성하십시오.
- perspective_differences:
  관점에 따라 평가가 달라지는 지점을 담은 문자열 목록
- tradeoffs:
  메모리 절감, 정확도, 전송 오버헤드, 처리량,
  적용 복잡성 등의 상충 관계 목록
- application_conditions:
  기술별로 적합하거나 제약이 발생하는 적용 조건 목록
- unresolved_gaps:
  공개 자료만으로 판단하기 어려운 근거 공백 목록
- cited_evidence_ids:
  종합 판단에 실제 사용한 valid_evidence_ids 목록
"""

    structured_llm = _get_llm().with_structured_output(Synthesis)
    raw_synthesis = structured_llm.invoke(prompt)

    if not isinstance(raw_synthesis, dict):
        raise TypeError("LLM이 Synthesis 형식의 결과를 반환하지 않았습니다.")

    synthesis = _validate_synthesis(
        synthesis=raw_synthesis,
        valid_evidence_ids=valid_evidence_ids,
        evidence_gaps=evidence_gaps,
    )

    return {"synthesis": synthesis}