"""State → 근거 준비 → 도메인 평가 → 부분 결과 병합의 실행 순서를 연결한다."""

from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.common.tasks import (
    criteria,
    domain_references,
    merge_evaluation,
)
from kv_cache_eval.common.tasks import technologies as target_technologies
from kv_cache_eval.features.domain.assessment import (
    DEFAULT_MAX_INPUT_BYTES,
    assess_payload,
    plan_requests,
)
from kv_cache_eval.features.domain.evidence import collect_evidence, has_text
from kv_cache_eval.features.domain.results import (
    DomainResponseError,
    build_result,
    invalid_response,
    unverified_evaluation,
)
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC
from kv_cache_eval.features.domain.runtime import (
    DomainConfigurationError,
    evaluation_settings,
)


def evaluate(
    state: State, *, max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
) -> StateUpdate:
    result = _evaluate_scope(state, max_input_bytes=max_input_bytes)
    if state.get("retry_request"):
        result["domain_eval"] = merge_evaluation(
            state, "domain_eval", result["domain_eval"]
        )
        result["domain_evidence"] = domain_references(state, result["domain_eval"])
    return result


@evaluation_settings()
def _evaluate_scope(
    state: State, *, max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
) -> StateUpdate:
    """근거 부족은 미확인으로 반환하고, 설정 오류는 재검색 대신 명시적으로 알린다."""
    selected_criteria = criteria(state, DOMAIN_RUBRIC)
    rows = {
        (tech, criterion): unverified_evaluation(
            tech, criterion, "확인된 평가 근거 없음"
        )
        for tech in target_technologies(state)
        for criterion in selected_criteria
    }
    notes = []
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise DomainConfigurationError("max_input_bytes는 양의 정수여야 합니다")
    # 입력 준비도 예외 처리 대상이다. TypedDict 자체는 실행 중 형식을 검사하지 않는다.
    try:
        by_id, payloads = _prepare_inputs(
            state, selected_criteria, rows, notes, max_input_bytes
        )
    except Exception as exc:
        if state.get("next_agent") == "domain":
            raise
        reason = (
            f"input_error: {type(exc).__name__}; 도메인 및 조사 결과 형식 확인 필요"
        )
        notes.append(reason)
        return build_result(
            {key: unverified_evaluation(*key, reason) for key in rows}, notes
        )

    for payload in payloads:
        try:
            assess_payload(payload, rows, notes, max_input_bytes)
        except DomainConfigurationError:
            raise
        except Exception as exc:
            if state.get("next_agent") == "domain":
                raise
            reason = f"execution_error: {type(exc).__name__}; 연결 또는 응답 확인 후 재실행 필요"
            notes.append(reason)
            for key in rows:
                if key[0] in payload["technologies"]:
                    rows[key] = invalid_response(*key, reason)
    unresolved = [
        key for key, row in rows.items() if row.get("failure_kind") == "response_error"
    ]
    if unresolved and state.get("next_agent") == "domain":
        raise DomainResponseError(f"도메인 응답 보정 소진: {len(unresolved)}개 항목")
    return build_result(rows, notes, by_id)


def _prepare_inputs(state, selected_criteria, rows, notes, max_input_bytes):
    """근거를 수집하고 미확인 항목을 표시한 뒤 평가 요청을 구성한다."""
    if not isinstance(state, dict):
        raise ValueError("state 형식 오류")
    settings = state.get("domain_and_criteria")
    if not isinstance(settings, dict) or not has_text(settings.get("domain")):
        raise ValueError("domain_and_criteria.domain 형식 오류")
    domain = settings["domain"]
    collected = collect_evidence(state)
    notes.extend(collected.notes)
    for tech, reason in collected.unavailable.items():
        for criterion in selected_criteria:
            rows[(tech, criterion)] = unverified_evaluation(tech, criterion, reason)
    for tech in collected.blocked:
        for criterion in selected_criteria:
            rows[(tech, criterion)] = unverified_evaluation(
                tech,
                criterion,
                "근거 충돌 또는 근거·제약사항 형식 오류로 해당 기술의 재조사 필요",
            )
    usable = {
        eid: item
        for eid, item in collected.by_id.items()
        if item["technology"] not in collected.blocked
    }
    if not usable:
        notes.append("evidence_missing: 사용 가능한 근거가 없어 LLM 호출 생략")
        return collected.by_id, []
    payloads = plan_requests(
        domain=domain,
        usable=usable,
        research_notes=collected.research_notes,
        selected_criteria=selected_criteria,
        rows=rows,
        notes=notes,
        limit=max_input_bytes,
        strict=state.get("next_agent") == "domain",
    )
    return collected.by_id, payloads
