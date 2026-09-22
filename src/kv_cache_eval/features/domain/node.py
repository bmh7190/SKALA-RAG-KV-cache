"""근거 수집 → 평가 초안 → 근거 검토 → 수치 채점. 인용 근거와 평가를 반환한다."""

import json
from copy import deepcopy

from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.features.domain.prompt import (
    OUTPUT_SCHEMA, SYSTEM_PROMPT, VERIFY_PROMPT, VERIFY_SCHEMA,
)
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC, NUMERIC_CRITERIA
from kv_cache_eval.features.domain.measurement import (
    validate_measurement, score_measurement, format_change, quote_exists,
)
from kv_cache_eval.features.domain.runtime import DomainConfigurationError, invoke_structured, evaluation_settings

TECHNOLOGIES = ("KIVI", "InfiniGen")
# 양쪽 요청(초안/검토)의 UTF-8 바이트 수를 각각 제한한다. 모델 한도에 맞게 조절 가능하다.
DEFAULT_MAX_INPUT_BYTES = 32000
MAX_REVIEW_DRAFT_BYTES = 4000
ALIASES = {"GPU 메모리": "GPU 메모리 사용량", "지연시간": "추론 지연시간", "운영 난이도": "적용·운영 난이도"}


def _utf8(value):
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
        return True
    except UnicodeError:
        return False


def _text(value):
    return _utf8(value) and bool(value.strip())


def _check_json(value):
    """재귀 직렬화·복사 전에 반복문으로 깊이와 구조 크기를 제한한다."""
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if depth > 32 or visited > 10000:
            raise ValueError("중첩 깊이 또는 구조 크기 초과")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend((child, depth + 1) for child in item)
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unknown(technology, criterion, reason):
    return {"technology": technology, "criterion": criterion, "judgment": None,
            "score": None, "rationale": None, "evidence_ids": [],
            "uncertainty": reason, "basis_status": "unverified"}


def _finish(rows, notes, by_id=None):
    # 보고서 담당자가 근거 부족과 점수 보류를 구분할 수 있도록 항목명을 남긴다.
    summary = []
    missing = [f"{tech}/{criterion}" for (tech, criterion), item in rows.items()
               if item["basis_status"] == "unverified"]
    pending_scores = [f"{tech}/{criterion}" for (tech, criterion), item in rows.items()
                      if item["basis_status"] != "unverified" and item["score"] is None]
    if missing:
        summary.append("근거·판단 미확인 항목: " + ", ".join(missing))
    if pending_scores:
        summary.append("판단 근거는 있으나 점수 미확인인 항목: " + ", ".join(pending_scores))
    evaluations = [rows[(tech, criterion)] for tech in TECHNOLOGIES for criterion in DOMAIN_RUBRIC]
    cited_ids = dict.fromkeys(eid for item in evaluations if item["basis_status"] != "unverified"
                              for eid in item["evidence_ids"])
    evidence = [by_id[eid] for eid in cited_ids if by_id is not None and eid in by_id]
    paragraphs = []
    for tech in TECHNOLOGIES:
        confirmed = [item for item in evaluations
                     if item["technology"] == tech and item["basis_status"] != "unverified"]
        if not confirmed:
            paragraphs.append(f"{tech}: 확인된 평가 근거가 없어 판단을 보류했습니다.")
            continue
        details = []
        for item in confirmed:
            score = f"{item['score']}점" if item["score"] is not None else "점수 미확인"
            details.append(f"{item['criterion']}({score}): {item['judgment']} {item['rationale']}")
        paragraph = f"{tech}: " + " ".join(details)
        if len(confirmed) < len(DOMAIN_RUBRIC):
            paragraph += f" 나머지 {len(DOMAIN_RUBRIC) - len(confirmed)}개 항목은 판단을 보류했습니다."
        paragraphs.append(paragraph)
    return {"domain_evidence": {
        "evidence": evidence,
        "notes": ["도메인 평가에서 실제 인용한 기술 조사 근거를 원래 ID와 출처로 재사용함"],
    }, "domain_eval": {
        "evaluations": evaluations,
        "notes": list(dict.fromkeys([*notes, *summary])),
        "text": "\n\n".join(paragraphs),
    }}


def _messages(system, payload):
    return [("system", system), ("human", json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))]


def _size(messages):
    # 빈 messages 배열을 실제 메시지로 교체한 크기를 계산한다.
    # JSON 이스케이프와 응답 스키마를 포함해 기존 크기 제한을 유지한다.
    message_bytes = len(json.dumps(
        [{"role": role, "content": text} for role, text in messages],
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8"))
    envelope_bytes = max(len(json.dumps({
        "messages": [],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": schema["title"], "strict": True, "schema": schema,
        }},
    }, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        for schema in (OUTPUT_SCHEMA, VERIFY_SCHEMA))
    return envelope_bytes - len(b"[]") + message_bytes


def _normalize_evidence(item):
    """선택 항목의 생략과 null을 통일하고 값이 있으면 형식을 검사한다."""
    _check_json(item)
    normalized = deepcopy(item)
    excerpt = normalized.setdefault("excerpt", None)
    if excerpt is not None and not _utf8(excerpt):
        raise ValueError("발췌 형식 오류")
    source = normalized["source"]
    for name in ("document", "url"):
        value = source.setdefault(name, None)
        if value is not None and not _utf8(value):
            raise ValueError("출처 형식 오류")
    page = source.setdefault("page", None)
    if page is not None and (type(page) is not int or page < 1):
        raise ValueError("페이지는 양의 정수여야 합니다")
    experiment = normalized.setdefault("experiment", None)
    if experiment is not None:
        if not isinstance(experiment, dict):
            raise ValueError("실험 정보 형식 오류")
        for name in ("model", "baseline", "workload"):
            value = experiment.setdefault(name, None)
            if value is not None and not _utf8(value):
                raise ValueError("실험 정보 값은 문자열이어야 합니다")
    return normalized


def _collect(state):
    """미확인 근거 제외 후 중복을 처리한다. 상충하는 동일 ID는 전부 제외한다."""
    by_id, conflicts, research_notes, notes = {}, set(), {}, []
    blocked, unavailable = set(), {}
    for technology, key in (("KIVI", "kivi_evidence"), ("InfiniGen", "infinigen_evidence")):
        research = state.get(key)
        research_notes[technology] = []
        if research is None:
            unavailable[technology] = "조사 결과가 아직 제공되지 않았습니다. 기술 조사 결과가 필요합니다."
            notes.append(f"{technology}: {unavailable[technology]}")
            continue
        if not isinstance(research, dict) or not isinstance(research.get("evidence"), list):
            unavailable[technology] = "조사 결과의 evidence 목록 형식이 잘못되어 평가할 수 없습니다."
            notes.append(f"{technology}: 조사 결과 없음 또는 잘못된 입력 형식")
            continue
        supplied_notes = research.get("notes", [])
        if not isinstance(supplied_notes, list) or any(not _utf8(note) for note in supplied_notes):
            unavailable[technology] = "조사 주의사항의 형식이 잘못되어 평가를 보류합니다."
            notes.append(f"{technology}: 조사 주의사항 형식 오류, 해당 조사 제외")
            continue
        research_notes[technology] = [note for note in supplied_notes if _text(note)]
        notes.extend(f"{technology} 조사 주의사항: {note}" for note in research_notes[technology])
        if not research["evidence"]:
            unavailable[technology] = "조사 결과에 근거가 0건입니다. 해당 기술의 추가 검색이 필요합니다."
            continue
        for item in research["evidence"]:
            if not isinstance(item, dict):
                notes.append(f"{technology}: 잘못된 근거 형식 제외")
                continue
            source = item.get("source")
            if not (
                item.get("technology") == technology
                and item.get("verification_status") == "source_checked"
                and _text(item.get("id")) and _text(item.get("claim"))
                and isinstance(source, dict)
                and (_text(source.get("document")) or _text(source.get("url")))
            ):
                notes.append(f"{technology}: 미확인 또는 불완전한 근거 제외")
                continue
            eid = item["id"]
            limitations = item.get("limitations", [])
            if not isinstance(limitations, list) or any(not _utf8(value) for value in limitations):
                blocked.add(technology)
                notes.append(f"{technology}: 제약사항 형식 오류; 해당 기술 평가 보류")
                continue
            try:
                item = _normalize_evidence(item)
            except (TypeError, ValueError, UnicodeError, RecursionError):
                blocked.add(technology)
                notes.append(f"{technology}: 근거의 중첩·문자·출처·실험 정보 형식 오류; 해당 기술 평가 보류")
                continue
            if eid in conflicts:
                blocked.add(technology)
                continue
            if eid in by_id:
                if item == by_id[eid]:
                    notes.append(f"중복 근거 {eid}: 동일 내용은 한 번만 사용")
                elif ({key: value for key, value in item.items() if key != "limitations"}
                      == {key: value for key, value in by_id[eid].items() if key != "limitations"}):
                    by_id[eid]["limitations"] = list(dict.fromkeys(by_id[eid].get("limitations", []) + limitations))
                    notes.append(f"중복 근거 {eid}: 같은 사실에 추가된 제약사항을 통합")
                else:
                    blocked.update((technology, by_id[eid]["technology"]))
                    conflicts.add(eid)
                    del by_id[eid]
                    notes.append(f"중복 근거 {eid}: 내용이 달라 모두 제외, 재조사 필요")
            else:
                by_id[eid] = item
    for technology in blocked:
        warning = "근거 충돌 또는 근거·제약사항 형식 오류가 있어 재조사 전 해당 기술 평가를 보류합니다."
        notes.append(f"{technology}: {warning}")
    for technology in TECHNOLOGIES:
        if technology not in blocked and not any(item["technology"] == technology for item in by_id.values()):
            unavailable.setdefault(technology, "사용할 수 있는 근거가 없습니다. 출처 확인 상태·근거 ID·내용을 확인해야 합니다.")
    return by_id, research_notes, notes, blocked, unavailable


def _supported_ids(item, technology, by_id):
    supports = item.get("supports")
    if not isinstance(supports, list) or not supports:
        return None
    ids = []
    for support in supports:
        if not isinstance(support, dict):
            return None
        eid = support.get("evidence_id")
        if not isinstance(eid, str) or eid not in by_id:
            return None
        evidence = by_id[eid]
        if evidence["technology"] != technology or not quote_exists(support.get("quote"), evidence):
            return None
        if eid not in ids:
            ids.append(eid)
    return ids


def _candidates(raw, by_id, rows, notes):
    if not isinstance(raw, dict) or not isinstance(raw.get("evaluations"), list):
        raise ValueError("평가 응답 형식 오류")
    # LLM 메모는 미검증 진단 정보로 표시한다.
    if isinstance(raw.get("notes"), list):
        notes.extend(f"평가 초안 메모(미검증): {note}" for note in raw["notes"] if _text(note))
    candidates, seen = {}, set()
    allowed_technologies = {item["technology"] for item in by_id.values()}
    for item in raw["evaluations"]:
        if not isinstance(item, dict):
            continue
        tech, criterion = item.get("technology"), item.get("criterion")
        if not isinstance(tech, str) or not isinstance(criterion, str):
            continue
        if tech not in allowed_technologies:
            continue
        criterion = ALIASES.get(criterion, criterion)
        key = (tech, criterion)
        if key not in rows:
            notes.append("알 수 없는 평가 항목 제외")
            continue
        if key in seen:
            candidates.pop(key, None)
            rows[key] = _unknown(*key, "평가 항목이 중복되어 재확인 필요")
            continue
        seen.add(key)
        try:
            _check_json(item)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            rows[key] = _unknown(*key, "평가 응답에 저장할 수 없는 문자 또는 값이 있어 재작성 필요")
            continue
        if item.get("uncertainty") is not None and not _utf8(item["uncertainty"]):
            rows[key] = _unknown(*key, "평가 주의사항 형식 오류; 문자열 또는 null로 재작성 필요")
            continue
        if not _text(item.get("judgment")):
            rows[key] = _unknown(*key, item.get("uncertainty") if _text(item.get("uncertainty")) else "판단 근거 부족")
            continue
        ids = _supported_ids(item, tech, by_id)
        score = item.get("score")
        if not ids or not _text(item.get("rationale")) or (score is not None and (type(score) is not int or not 1 <= score <= 5)):
            rows[key] = _unknown(*key, "인용문·근거 ID·판단 이유·점수 형식 확인 필요")
            continue
        candidate = dict(item)
        candidate.update(criterion=criterion, evidence_ids=ids)
        if criterion in NUMERIC_CRITERIA and item.get("measurement") is not None:
            if not validate_measurement(item["measurement"], ids, by_id):
                rows[key] = _unknown(*key, "원문 조건과 인용문에서 확정적인 비교 측정값을 확인할 수 없음; 범위·근삿값·한계·부호 확인 필요")
                continue
        candidates[key] = candidate
    for key in rows:
        if key[0] in allowed_technologies and key not in seen:
            rows[key] = _unknown(*key, "평가 응답에 이 항목이 누락되었습니다. 해당 항목의 평가를 다시 작성해야 합니다.")
    return candidates


def _apply_reviews(raw, candidates, by_id, research_notes, rows):
    if not isinstance(raw, dict) or not isinstance(raw.get("reviews"), list):
        raise ValueError("근거 검토 응답 형식 오류")
    reviews, duplicates = {}, set()
    for review in raw["reviews"]:
        if not isinstance(review, dict):
            continue
        tech, criterion = review.get("technology"), review.get("criterion")
        if not isinstance(tech, str) or not isinstance(criterion, str):
            continue
        key = (tech, ALIASES.get(criterion, criterion))
        if key in reviews:
            duplicates.add(key)
        reviews[key] = review
    for key, candidate in candidates.items():
        review = reviews.get(key, {})
        reason = review.get("reason")
        if key in duplicates or review.get("supported") is not True or not _text(reason):
            rows[key] = _unknown(*key, f"근거 검토 미통과: {reason if _text(reason) else '검토 결과 없음 또는 중복'}")
            continue
        ids = candidate["evidence_ids"]
        score, judgment, rationale = candidate.get("score"), candidate["judgment"], candidate["rationale"]
        uncertainties = [candidate.get("uncertainty"), "LLM 근거 검토를 거친 추론이며 실제 운영 검증이 필요합니다."]
        if key[1] in NUMERIC_CRITERIA:
            measurement = candidate.get("measurement")
            if measurement is None:
                score = None
                uncertainties.append("비교 가능한 측정값이 없어 정량 점수 미확인")
            elif review.get("measurement_supported") is not True:
                rows[key] = _unknown(*key, f"측정값의 지표·단위·비교 조건 확인 실패: {reason}")
                continue
            else:
                try:
                    score, change, detail = score_measurement(key[1], measurement)
                    displayed_change = format_change(change)
                except (ArithmeticError, ValueError, TypeError):
                    rows[key] = _unknown(*key, "측정값 계산 실패: 해당 항목의 수치와 계산 조건 확인 필요")
                    continue
                direction = "증가율" if key[1] == "처리량" else "감소율"
                judgment = f"인용 실험의 {key[1]} {direction}: {displayed_change}%"
                rationale = (
                    detail + "; "
                    f"{direction} {displayed_change}%. "
                    + (f"{score}점 기준: {DOMAIN_RUBRIC[key[1]]['scores'][score]}" if score is not None else "원문 점수 구간이 겹침")
                )
                if score is None:
                    uncertainties.append("측정 사실은 확인됨. 원문 점수 경계가 겹쳐 점수만 미확인; 팀 기준 확정 필요")
        elif score is not None and review.get("rubric_supported") is not True:
            rows[key] = _unknown(*key, f"정성 점수와 Rubric의 대응 확인 실패: {reason}")
            continue
        elif score is None:
            uncertainties.append("정성 판단은 가능하지만 수치 점수는 미확인")
        uncertainties.extend(research_notes.get(key[0], []))
        for eid in ids:
            experiment = by_id[eid].get("experiment")
            if isinstance(experiment, dict):
                context = "; ".join(f"{field}={experiment[field]}" for field in ("model", "baseline", "workload")
                                    if _text(experiment.get(field)))
            else:
                context = ""
            if context:
                rationale += f" / 실험 조건 [{eid}]: {context}"
            else:
                uncertainties.append(f"{eid}: 실험 조건 메타데이터 미확인")
            uncertainties.extend(by_id[eid].get("limitations", []))
        rows[key] = {"technology": key[0], "criterion": key[1], "judgment": judgment,
                     "score": score, "rationale": rationale, "evidence_ids": ids,
                     "uncertainty": " / ".join(dict.fromkeys(x for x in uncertainties if _text(x))),
                     "basis_status": "inferred"}


def _payload(domain, technologies, by_id, research_notes):
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
    return {"domain": domain, "technologies": list(technologies), "rubric": DOMAIN_RUBRIC,
            "research_notes": {tech: research_notes[tech] for tech in technologies},
            "evidence": evidence}


def _fits(payload, limit):
    # JSON 문자열이 메시지에 들어갈 때 따옴표·역슬래시가 최대 두 배로 늘어난다.
    # 배열 구분자 여유까지 포함해 한 항목을 검토할 공간을 확보한다.
    return (_size(_messages(SYSTEM_PROMPT, payload)) <= limit
            and _size(_messages(VERIFY_PROMPT, {**payload, "drafts": []}))
            + 2 * MAX_REVIEW_DRAFT_BYTES + 2 <= limit)


def _review_draft(candidate):
    # 근거 ID는 supports에 이미 있다. 내부 집계용 중복 목록은 전송하지 않는다.
    return {name: value for name, value in candidate.items() if name != "evidence_ids"}


def _review_batches(payload, candidates, limit, rows, notes):
    """각 검토 요청에 전체 비교 근거를 유지하고 초안만 나누어 보낸다."""
    batch = {}
    for key, candidate in candidates.items():
        review_draft = _review_draft(candidate)
        if len(json.dumps(review_draft, ensure_ascii=False).encode("utf-8")) > MAX_REVIEW_DRAFT_BYTES:
            rows[key] = _unknown(*key, "평가 초안이 항목별 검토 한도를 초과함; 짧은 인용문으로 재작성 필요")
            notes.append(f"draft_budget_exceeded: {key[0]} / {key[1]}")
            continue
        proposed = {**batch, key: candidate}
        messages = _messages(VERIFY_PROMPT, {**payload, "drafts": [_review_draft(item) for item in proposed.values()]})
        if _size(messages) <= limit:
            batch = proposed
        else:
            if batch:
                yield batch
            batch = {key: candidate}
            # 예약한 공간을 넘어서는 값은 전송 직전에도 차단한다.
            if _size(_messages(VERIFY_PROMPT, {**payload, "drafts": [review_draft]})) > limit:
                rows[key] = _unknown(*key, "근거 검토 요청이 입력 한도 초과")
                notes.append(f"input_budget_exceeded: {key[0]} / {key[1]}")
                batch = {}
    if batch:
        yield batch


@evaluation_settings()
def evaluate(state: State, *, max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES) -> StateUpdate:
    """근거 부족은 미확인으로 반환하고, 설정 오류는 재검색 대신 명시적으로 알린다."""
    rows = {(tech, criterion): _unknown(tech, criterion, "확인된 평가 근거 없음")
            for tech in TECHNOLOGIES for criterion in DOMAIN_RUBRIC}
    notes = []
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise DomainConfigurationError("max_input_bytes는 양의 정수여야 합니다")
    # 입력 준비도 예외 처리 대상이다. TypedDict 자체는 실행 중 형식을 검사하지 않는다.
    try:
        if not isinstance(state, dict):
            raise ValueError("state 형식 오류")
        settings = state.get("domain_and_criteria")
        if not isinstance(settings, dict) or not _text(settings.get("domain")):
            raise ValueError("domain_and_criteria.domain 형식 오류")
        domain = settings["domain"]
        by_id, research_notes, notes, blocked, unavailable = _collect(state)
        for tech, reason in unavailable.items():
            for criterion in DOMAIN_RUBRIC:
                rows[(tech, criterion)] = _unknown(tech, criterion, reason)
        for tech in blocked:
            for criterion in DOMAIN_RUBRIC:
                rows[(tech, criterion)] = _unknown(tech, criterion, "근거 충돌 또는 근거·제약사항 형식 오류로 해당 기술의 재조사 필요")
        usable = {eid: item for eid, item in by_id.items() if item["technology"] not in blocked}
        if not usable:
            notes.append("evidence_missing: 사용 가능한 근거가 없어 LLM 호출 생략")
            return _finish(rows, notes, by_id)
        technologies = [tech for tech in TECHNOLOGIES if any(item["technology"] == tech for item in usable.values())]
        combined = _payload(domain, technologies, usable, research_notes)
        if _fits(combined, max_input_bytes):
            payloads = [combined]
        else:
            # 한 기술의 자료량이 다른 기술의 공간을 차지하지 않도록 독립적으로 배정한다.
            payloads = []
            for tech in technologies:
                payload = _payload(domain, [tech], usable, research_notes)
                if _fits(payload, max_input_bytes):
                    payloads.append(payload)
                else:
                    notes.append(f"input_budget_exceeded: {tech} 전체 근거와 검토 여유분이 한도 초과")
                    for criterion in DOMAIN_RUBRIC:
                        rows[(tech, criterion)] = _unknown(tech, criterion, "전체 근거가 입력 한도 초과; 관련 문서 범위 축소 또는 한도 조정 필요")
    except Exception as exc:
        reason = f"input_error: {type(exc).__name__}; 도메인 및 조사 결과 형식 확인 필요"
        notes.append(reason)
        return _finish({key: _unknown(*key, reason) for key in rows}, notes)

    for payload in payloads:
        selected = {item["id"]: item for item in payload["evidence"]}
        try:
            raw = invoke_structured(_messages(SYSTEM_PROMPT, payload), OUTPUT_SCHEMA)
            candidates = _candidates(raw, selected, rows, notes)
        except DomainConfigurationError:
            # 공용 그래프는 미확인을 근거 부족으로 판단하므로 설정 오류를 그 경로로 보내지 않는다.
            raise
        except Exception as exc:
            reason = f"execution_error (assessment): {type(exc).__name__}; 연결 또는 응답 확인 후 재실행 필요"
            notes.append(reason)
            for key in rows:
                if key[0] in payload["technologies"]:
                    rows[key] = _unknown(*key, reason)
            continue
        for batch in _review_batches(payload, candidates, max_input_bytes, rows, notes):
            try:
                reviews = invoke_structured(
                    _messages(VERIFY_PROMPT, {**payload, "drafts": [_review_draft(item) for item in batch.values()]}),
                    VERIFY_SCHEMA,
                )
                _apply_reviews(reviews, batch, selected, research_notes, rows)
            except DomainConfigurationError:
                raise
            except Exception as exc:
                reason = f"execution_error (verification): {type(exc).__name__}; 검토 응답 확인 후 재실행 필요"
                notes.append(reason)
                for key in batch:
                    rows[key] = _unknown(*key, reason)
    return _finish(rows, notes, by_id)
