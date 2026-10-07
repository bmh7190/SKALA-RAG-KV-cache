"""입력 예산 안에서 초안 → 근거 검토 → 오류 항목 1회 보정을 실행한다."""

import json

from kv_cache_eval.common.errors import InputBudgetExceeded
from kv_cache_eval.features.domain.prompt import (
    OUTPUT_SCHEMA,
    REPAIR_INSTRUCTIONS,
    SYSTEM_PROMPT,
    VERIFY_PROMPT,
    VERIFY_SCHEMA,
    repair_schema,
)
from kv_cache_eval.features.domain.results import (
    invalid_response,
    unverified_evaluation,
)
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC
from kv_cache_eval.features.domain.runtime import invoke_structured
from kv_cache_eval.features.domain.validation import apply_reviews, validate_drafts

TECHNOLOGIES = ("KIVI", "InfiniGen")
DEFAULT_MAX_INPUT_BYTES = 131072
MAX_REVIEW_DRAFT_BYTES = 4000


def plan_requests(
    *, domain, usable, research_notes, selected_criteria, rows, notes, limit, strict
):
    """전체 근거를 한 요청에 넣고, 한도 초과 시 기술별로 나눈다."""
    technologies = [
        tech
        for tech in TECHNOLOGIES
        if any(item["technology"] == tech for item in usable.values())
    ]
    combined = build_payload(domain, technologies, usable, research_notes)
    combined["rubric"] = {key: DOMAIN_RUBRIC[key] for key in selected_criteria}
    if fits_input_budget(combined, limit):
        return [combined]

    # 한 기술의 자료량이 다른 기술의 공간을 차지하지 않도록 독립적으로 배정한다.
    payloads = []
    for tech in technologies:
        payload = build_payload(domain, [tech], usable, research_notes)
        payload["rubric"] = combined["rubric"]
        if fits_input_budget(payload, limit):
            payloads.append(payload)
            continue
        if strict:
            raise InputBudgetExceeded("평가 근거가 입력 한도를 초과했습니다")
        notes.append(
            f"input_budget_exceeded: {tech} 전체 근거와 검토 여유분이 한도 초과"
        )
        for criterion in selected_criteria:
            rows[(tech, criterion)] = unverified_evaluation(
                tech,
                criterion,
                "전체 근거가 입력 한도 초과; 관련 문서 범위 축소 또는 한도 조정 필요",
            )
    return payloads


def assess_payload(payload, rows, notes, limit):
    """초안을 검토하고 응답 오류가 난 항목만 같은 근거로 한 번 보정한다."""
    selected = {item["id"]: item for item in payload["evidence"]}
    active = {
        key: value
        for key, value in rows.items()
        if key[0] in payload["technologies"] and key[1] in payload["rubric"]
    }
    raw = invoke_structured(request_messages(SYSTEM_PROMPT, payload), OUTPUT_SCHEMA)
    _review_candidates(raw, payload, active, notes, limit)
    rows.update(active)
    failed = {
        key: row
        for key, row in active.items()
        if row.get("failure_kind") == "response_error"
    }
    if not failed:
        return
    notes.append(f"response_repair: {len(failed)}개 항목을 같은 근거로 각각 1회 재작성")
    for (technology, criterion), row in failed.items():
        key = (technology, criterion)
        repaired = _repair_evaluation(
            payload, raw, key, row["uncertainty"], selected, limit
        )
        active = {key: row}
        _review_candidates(repaired, payload, active, notes, limit)
        rows.update(active)
    remaining = sum(rows[key].get("failure_kind") == "response_error" for key in failed)
    if remaining:
        notes.append(f"response_repair_exhausted: {remaining}개 항목 응답 오류")


def request_messages(system, payload):
    return [
        ("system", system),
        (
            "human",
            json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ),
        ),
    ]


def request_size(messages):
    # 빈 messages 배열을 실제 메시지로 교체한 크기를 계산한다.
    # JSON 이스케이프와 응답 스키마를 포함해 기존 크기 제한을 유지한다.
    message_bytes = len(
        json.dumps(
            [{"role": role, "content": text} for role, text in messages],
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    envelope_bytes = max(
        len(
            json.dumps(
                {
                    "messages": [],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema["title"],
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        for schema in (OUTPUT_SCHEMA, VERIFY_SCHEMA)
    )
    return envelope_bytes - len(b"[]") + message_bytes


def build_payload(domain, technologies, by_id, research_notes):
    # 조사 노드가 claim과 excerpt에 같은 원문을 넣는 경우 한 번만 전송한다.
    # 서로 다른 발췌와 제약사항은 유지하고 입력 State는 변경하지 않는다.
    evidence = []
    for item in by_id.values():
        if item["technology"] not in technologies:
            continue
        entry = dict(item)
        if entry.get("excerpt") == entry["claim"]:
            entry.pop("excerpt")
        evidence.append(entry)
    return {
        "domain": domain,
        "technologies": list(technologies),
        "rubric": DOMAIN_RUBRIC,
        "research_notes": {tech: research_notes[tech] for tech in technologies},
        "evidence": evidence,
    }


def fits_input_budget(payload, limit):
    # JSON 문자열이 메시지에 들어갈 때 따옴표·역슬래시가 최대 두 배로 늘어난다.
    # 배열 구분자 여유까지 포함해 한 항목을 검토할 공간을 확보한다.
    return (
        request_size(request_messages(SYSTEM_PROMPT, payload)) <= limit
        and request_size(request_messages(VERIFY_PROMPT, {**payload, "drafts": []}))
        + 2 * MAX_REVIEW_DRAFT_BYTES
        + 2
        <= limit
    )


def _review_draft(candidate):
    # 근거 ID는 supports에 이미 있다. 내부 집계용 중복 목록은 전송하지 않는다.
    return {name: value for name, value in candidate.items() if name != "evidence_ids"}


def _review_batches(payload, candidates, limit, rows, notes):
    """각 검토 요청에 전체 비교 근거를 유지하고 초안만 나누어 보낸다."""
    batch = {}
    for key, candidate in candidates.items():
        review_draft = _review_draft(candidate)
        if (
            len(json.dumps(review_draft, ensure_ascii=False).encode("utf-8"))
            > MAX_REVIEW_DRAFT_BYTES
        ):
            rows[key] = invalid_response(
                *key,
                "평가 초안이 항목별 검토 한도를 초과함; 짧은 인용문으로 재작성 필요",
            )
            notes.append(f"draft_budget_exceeded: {key[0]} / {key[1]}")
            continue
        proposed = {**batch, key: candidate}
        messages = request_messages(
            VERIFY_PROMPT,
            {**payload, "drafts": [_review_draft(item) for item in proposed.values()]},
        )
        if request_size(messages) <= limit:
            batch = proposed
        else:
            if batch:
                yield batch
            batch = {key: candidate}
            # 예약한 공간을 넘어서는 값은 전송 직전에도 차단한다.
            if (
                request_size(
                    request_messages(
                        VERIFY_PROMPT, {**payload, "drafts": [review_draft]}
                    )
                )
                > limit
            ):
                rows[key] = invalid_response(*key, "근거 검토 요청이 입력 한도 초과")
                notes.append(f"input_budget_exceeded: {key[0]} / {key[1]}")
                batch = {}
    if batch:
        yield batch


def _review_candidates(raw, payload, active, notes, limit):
    selected = {item["id"]: item for item in payload["evidence"]}
    candidates = validate_drafts(raw, selected, active, notes)
    for batch in _review_batches(payload, candidates, limit, active, notes):
        reviews = invoke_structured(
            request_messages(
                VERIFY_PROMPT,
                {
                    **payload,
                    "drafts": [_review_draft(item) for item in batch.values()],
                },
            ),
            VERIFY_SCHEMA,
        )
        apply_reviews(reviews, batch, selected, payload["research_notes"], active)


def _restore_quotes(raw, evidence):
    """Resolve source selectors without inventing or rewriting any quoted text."""
    for item in raw.get("evaluations", []):
        for support in item.get("supports", []):
            field = support.pop("quote_field", None)
            source = evidence.get(support.get("evidence_id"), {})
            support["quote"] = (
                source.get(field, source.get("claim"))
                if field in ("claim", "excerpt")
                else None
            )
    return raw


def _repair_evaluation(payload, raw, key, reason, selected, limit):
    """잘못된 한 항목만 같은 근거로 다시 작성하고 원문 인용을 복원한다."""
    technology, criterion = key
    request = {
        **payload,
        "technologies": [technology],
        "rubric": {criterion: DOMAIN_RUBRIC[criterion]},
        "repair_targets": [
            {
                "technology": technology,
                "criterion": criterion,
                "reason": reason,
            }
        ],
        "previous_drafts": [
            item
            for item in raw["evaluations"]
            if isinstance(item, dict)
            and (item.get("technology"), item.get("criterion"))
            == (technology, criterion)
        ],
    }
    system = SYSTEM_PROMPT + REPAIR_INSTRUCTIONS
    if request_size(request_messages(system, request)) > limit:
        raise InputBudgetExceeded("도메인 응답 보정 요청이 입력 한도를 초과했습니다")
    repaired = invoke_structured(
        request_messages(system, request),
        repair_schema(technology, criterion, selected),
    )
    return _restore_quotes(repaired, selected)
