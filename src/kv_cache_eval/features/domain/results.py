"""미확인 평가와 최종 State 갱신값을 만든다. 모델을 호출하지 않는다."""

from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC


class DomainResponseError(ValueError):
    """Bounded rewriting could not produce a verifiable evaluation response."""


def unverified_evaluation(
    technology, criterion, reason, *, failure_kind="evidence_gap"
):
    return {
        "technology": technology,
        "criterion": criterion,
        "judgment": None,
        "score": None,
        "rationale": None,
        "evidence_ids": [],
        "uncertainty": reason,
        "basis_status": "unverified",
        "failure_kind": failure_kind,
    }


def invalid_response(technology, criterion, reason):
    return unverified_evaluation(
        technology, criterion, reason, failure_kind="response_error"
    )


def build_result(rows, notes, by_id=None):
    # 보고서 담당자가 근거 부족과 점수 보류를 구분할 수 있도록 항목명을 남긴다.
    summary = []
    missing = [
        f"{tech}/{criterion}"
        for (tech, criterion), item in rows.items()
        if item["basis_status"] == "unverified"
    ]
    pending_scores = [
        f"{tech}/{criterion}"
        for (tech, criterion), item in rows.items()
        if item["basis_status"] != "unverified" and item["score"] is None
    ]
    if missing:
        summary.append("근거·판단 미확인 항목: " + ", ".join(missing))
    if pending_scores:
        summary.append(
            "판단 근거는 있으나 점수 미확인인 항목: " + ", ".join(pending_scores)
        )
    evaluations = list(rows.values())
    cited_ids = dict.fromkeys(
        eid
        for item in evaluations
        if item["basis_status"] != "unverified"
        for eid in item["evidence_ids"]
    )
    evidence = [by_id[eid] for eid in cited_ids if by_id is not None and eid in by_id]
    paragraphs = []
    for tech in dict.fromkeys(key[0] for key in rows):
        confirmed = [
            item
            for item in evaluations
            if item["technology"] == tech and item["basis_status"] != "unverified"
        ]
        if not confirmed:
            paragraphs.append(f"{tech}: 확인된 평가 근거가 없어 판단을 보류했습니다.")
            continue
        details = []
        for item in confirmed:
            score = f"{item['score']}점" if item["score"] is not None else "점수 미확인"
            details.append(
                f"{item['criterion']}({score}): {item['judgment']} {item['rationale']}"
            )
        paragraph = f"{tech}: " + " ".join(details)
        if len(confirmed) < len(DOMAIN_RUBRIC):
            paragraph += f" 나머지 {len(DOMAIN_RUBRIC) - len(confirmed)}개 항목은 판단을 보류했습니다."
        paragraphs.append(paragraph)
    return {
        "domain_evidence": {
            "evidence": evidence,
            "notes": [
                "도메인 평가에서 실제 인용한 기술 조사 근거를 원래 ID와 출처로 재사용함"
            ],
        },
        "domain_eval": {
            "evaluations": evaluations,
            "notes": list(dict.fromkeys([*notes, *summary])),
            "text": "\n\n".join(paragraphs),
        },
    }
