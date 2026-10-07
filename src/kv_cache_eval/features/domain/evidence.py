"""조사 근거의 출처·형식·중복을 검사하고 평가에 쓸 근거를 수집한다."""

import json
from copy import deepcopy
from dataclasses import dataclass

from kv_cache_eval.common.schemas import Evidence
from kv_cache_eval.common.tasks import technologies as target_technologies


@dataclass
class CollectedEvidence:
    """순서에 의존하는 튜플 대신 수집 결과를 이름으로 구분한다."""

    by_id: dict[str, Evidence]
    research_notes: dict[str, list[str]]
    notes: list[str]
    blocked: set[str]
    unavailable: dict[str, str]


def is_utf8(value):
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
        return True
    except UnicodeError:
        return False


def has_text(value):
    return is_utf8(value) and bool(value.strip())


def validate_json_structure(value):
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


def _normalize_evidence(item):
    """선택 항목의 생략과 null을 통일하고 값이 있으면 형식을 검사한다."""
    validate_json_structure(item)
    normalized = deepcopy(item)
    excerpt = normalized.setdefault("excerpt", None)
    if excerpt is not None and not is_utf8(excerpt):
        raise ValueError("발췌 형식 오류")
    source = normalized["source"]
    for name in ("document", "url"):
        value = source.setdefault(name, None)
        if value is not None and not is_utf8(value):
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
            if value is not None and not is_utf8(value):
                raise ValueError("실험 정보 값은 문자열이어야 합니다")
    return normalized


def collect_evidence(state) -> CollectedEvidence:
    """미확인 근거 제외 후 중복을 처리한다. 상충하는 동일 ID는 전부 제외한다."""
    by_id, conflicts, research_notes, notes = {}, set(), {}, []
    blocked, unavailable = set(), {}
    for technology, key in (
        ("KIVI", "kivi_evidence"),
        ("InfiniGen", "infinigen_evidence"),
    ):
        if technology not in target_technologies(state):
            continue
        research = state.get(key)
        research_notes[technology] = []
        if research is None:
            unavailable[technology] = (
                "조사 결과가 아직 제공되지 않았습니다. 기술 조사 결과가 필요합니다."
            )
            notes.append(f"{technology}: {unavailable[technology]}")
            continue
        if not isinstance(research, dict) or not isinstance(
            research.get("evidence"), list
        ):
            unavailable[technology] = (
                "조사 결과의 evidence 목록 형식이 잘못되어 평가할 수 없습니다."
            )
            notes.append(f"{technology}: 조사 결과 없음 또는 잘못된 입력 형식")
            continue
        supplied_notes = research.get("notes", [])
        if not isinstance(supplied_notes, list) or any(
            not is_utf8(note) for note in supplied_notes
        ):
            unavailable[technology] = (
                "조사 주의사항의 형식이 잘못되어 평가를 보류합니다."
            )
            notes.append(f"{technology}: 조사 주의사항 형식 오류, 해당 조사 제외")
            continue
        research_notes[technology] = [note for note in supplied_notes if has_text(note)]
        notes.extend(
            f"{technology} 조사 주의사항: {note}" for note in research_notes[technology]
        )
        if not research["evidence"]:
            unavailable[technology] = (
                "조사 결과에 근거가 0건입니다. 해당 기술의 추가 검색이 필요합니다."
            )
            continue
        for item in research["evidence"]:
            if not isinstance(item, dict):
                notes.append(f"{technology}: 잘못된 근거 형식 제외")
                continue
            source = item.get("source")
            if not (
                item.get("technology") == technology
                and item.get("verification_status") == "source_checked"
                and has_text(item.get("id"))
                and has_text(item.get("claim"))
                and isinstance(source, dict)
                and (has_text(source.get("document")) or has_text(source.get("url")))
            ):
                notes.append(f"{technology}: 미확인 또는 불완전한 근거 제외")
                continue
            eid = item["id"]
            limitations = item.get("limitations", [])
            if not isinstance(limitations, list) or any(
                not is_utf8(value) for value in limitations
            ):
                blocked.add(technology)
                notes.append(f"{technology}: 제약사항 형식 오류; 해당 기술 평가 보류")
                continue
            try:
                item = _normalize_evidence(item)
            except (TypeError, ValueError, UnicodeError, RecursionError):
                blocked.add(technology)
                notes.append(
                    f"{technology}: 근거의 중첩·문자·출처·실험 정보 형식 오류; 해당 기술 평가 보류"
                )
                continue
            if eid in conflicts:
                blocked.add(technology)
                continue
            if eid in by_id:
                if item == by_id[eid]:
                    notes.append(f"중복 근거 {eid}: 동일 내용은 한 번만 사용")
                elif {
                    key: value for key, value in item.items() if key != "limitations"
                } == {
                    key: value
                    for key, value in by_id[eid].items()
                    if key != "limitations"
                }:
                    by_id[eid]["limitations"] = list(
                        dict.fromkeys(by_id[eid].get("limitations", []) + limitations)
                    )
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
    for technology in target_technologies(state):
        if technology not in blocked and not any(
            item["technology"] == technology for item in by_id.values()
        ):
            unavailable.setdefault(
                technology,
                "사용할 수 있는 근거가 없습니다. 출처 확인 상태·근거 ID·내용을 확인해야 합니다.",
            )
    return CollectedEvidence(
        by_id=by_id,
        research_notes=research_notes,
        notes=notes,
        blocked=blocked,
        unavailable=unavailable,
    )
