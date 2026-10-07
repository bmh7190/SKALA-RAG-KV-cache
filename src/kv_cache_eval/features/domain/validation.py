"""모델 초안의 인용·측정값을 검증하고 근거 검토 결과로 평가를 확정한다."""

from kv_cache_eval.features.domain.evidence import (
    has_text,
    is_utf8,
    validate_json_structure,
)
from kv_cache_eval.features.domain.measurement import (
    format_change,
    quote_exists,
    score_measurement,
    validate_measurement,
)
from kv_cache_eval.features.domain.results import (
    invalid_response,
    unverified_evaluation,
)
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC, NUMERIC_CRITERIA

ALIASES = {
    "GPU 메모리": "GPU 메모리 사용량",
    "지연시간": "추론 지연시간",
    "운영 난이도": "적용·운영 난이도",
}


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
        if evidence["technology"] != technology or not quote_exists(
            support.get("quote"), evidence
        ):
            return None
        if eid not in ids:
            ids.append(eid)
    return ids


def validate_drafts(raw, by_id, rows, notes):
    if not isinstance(raw, dict) or not isinstance(raw.get("evaluations"), list):
        raise ValueError("평가 응답 형식 오류")
    # LLM 메모는 미검증 진단 정보로 표시한다.
    if isinstance(raw.get("notes"), list):
        notes.extend(
            f"평가 초안 메모(미검증): {note}" for note in raw["notes"] if has_text(note)
        )
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
            rows[key] = invalid_response(*key, "평가 항목이 중복되어 재확인 필요")
            continue
        seen.add(key)
        try:
            validate_json_structure(item)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            rows[key] = invalid_response(
                *key, "평가 응답에 저장할 수 없는 문자 또는 값이 있어 재작성 필요"
            )
            continue
        if item.get("uncertainty") is not None and not is_utf8(item["uncertainty"]):
            rows[key] = invalid_response(
                *key, "평가 주의사항 형식 오류; 문자열 또는 null로 재작성 필요"
            )
            continue
        if not has_text(item.get("judgment")):
            rows[key] = unverified_evaluation(
                *key,
                item.get("uncertainty")
                if has_text(item.get("uncertainty"))
                else "판단 근거 부족",
            )
            continue
        ids = _supported_ids(item, tech, by_id)
        score = item.get("score")
        if (
            not ids
            or not has_text(item.get("rationale"))
            or (score is not None and (type(score) is not int or not 1 <= score <= 5))
        ):
            rows[key] = invalid_response(
                *key, "인용문·근거 ID·판단 이유·점수 형식 확인 필요"
            )
            continue
        candidate = dict(item)
        candidate.update(criterion=criterion, evidence_ids=ids)
        if criterion in NUMERIC_CRITERIA and item.get("measurement") is not None:
            if not validate_measurement(item["measurement"], ids, by_id):
                rows[key] = invalid_response(
                    *key,
                    "원문 조건과 인용문에서 확정적인 비교 측정값을 확인할 수 없음; 범위·근삿값·한계·부호 확인 필요",
                )
                continue
        candidates[key] = candidate
    for key in rows:
        if key[0] in allowed_technologies and key not in seen:
            rows[key] = invalid_response(
                *key,
                "평가 응답에 이 항목이 누락되었습니다. 해당 항목의 평가를 다시 작성해야 합니다.",
            )
    return candidates


def apply_reviews(raw, candidates, by_id, research_notes, rows):
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
        if (
            key in duplicates
            or type(review.get("supported")) is not bool
            or not has_text(reason)
        ):
            rows[key] = invalid_response(
                *key,
                f"근거 검토 미통과: {reason if has_text(reason) else '검토 결과 없음 또는 중복'}",
            )
            continue
        if not review["supported"]:
            rows[key] = unverified_evaluation(*key, f"근거 검토 미통과: {reason}")
            continue
        ids = candidate["evidence_ids"]
        score, judgment, rationale = (
            candidate.get("score"),
            candidate["judgment"],
            candidate["rationale"],
        )
        uncertainties = [
            candidate.get("uncertainty"),
            "LLM 근거 검토를 거친 추론이며 실제 운영 검증이 필요합니다.",
        ]
        if key[1] in NUMERIC_CRITERIA:
            measurement = candidate.get("measurement")
            if measurement is None:
                score = None
                uncertainties.append("비교 가능한 측정값이 없어 정량 점수 미확인")
            elif review.get("measurement_supported") is not True:
                rows[key] = invalid_response(
                    *key, f"측정값의 지표·단위·비교 조건 확인 실패: {reason}"
                )
                continue
            else:
                try:
                    score, change, detail = score_measurement(key[1], measurement)
                    displayed_change = format_change(change)
                except (ArithmeticError, ValueError, TypeError):
                    rows[key] = invalid_response(
                        *key, "측정값 계산 실패: 해당 항목의 수치와 계산 조건 확인 필요"
                    )
                    continue
                direction = "증가율" if key[1] == "처리량" else "감소율"
                judgment = f"인용 실험의 {key[1]} {direction}: {displayed_change}%"
                rationale = (
                    detail + "; "
                    f"{direction} {displayed_change}%. "
                    + (
                        f"{score}점 기준: {DOMAIN_RUBRIC[key[1]]['scores'][score]}"
                        if score is not None
                        else "원문 점수 구간이 겹침"
                    )
                )
                if score is None:
                    uncertainties.append(
                        "측정 사실은 확인됨. 원문 점수 경계가 겹쳐 점수만 미확인; 팀 기준 확정 필요"
                    )
        elif score is not None and review.get("rubric_supported") is not True:
            score = None
            uncertainties.append(f"정성 판단은 지지되지만 점수 기준은 미확인: {reason}")
        elif score is None:
            uncertainties.append("정성 판단은 가능하지만 수치 점수는 미확인")
        uncertainties.extend(research_notes.get(key[0], []))
        for eid in ids:
            experiment = by_id[eid].get("experiment")
            if isinstance(experiment, dict):
                context = "; ".join(
                    f"{field}={experiment[field]}"
                    for field in ("model", "baseline", "workload")
                    if has_text(experiment.get(field))
                )
            else:
                context = ""
            if context:
                rationale += f" / 실험 조건 [{eid}]: {context}"
            else:
                uncertainties.append(f"{eid}: 실험 조건 메타데이터 미확인")
            uncertainties.extend(by_id[eid].get("limitations", []))
        rows[key] = {
            "technology": key[0],
            "criterion": key[1],
            "judgment": judgment,
            "score": score,
            "rationale": rationale,
            "evidence_ids": ids,
            "uncertainty": " / ".join(
                dict.fromkeys(x for x in uncertainties if has_text(x))
            ),
            "basis_status": "inferred",
        }
