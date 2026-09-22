"""보고서와 종합에서 확인된 근거 ID만 인용하도록 돕는다."""

import re

from kv_cache_eval.common.schemas import Evidence
from kv_cache_eval.common.state import State


EVIDENCE_KEYS = ("kivi_evidence", "infinigen_evidence", "market_evidence", "domain_evidence")
_INLINE_CITATION = re.compile(r"\[([^\[\]\n]+)\]")


def collect_verified_evidence(state: State) -> dict[str, Evidence]:
    """확인된 원본 근거를 모은다. 도메인 노드의 동일 ID 재사용은 허용한다."""
    by_id: dict[str, Evidence] = {}
    for key in EVIDENCE_KEYS:
        result = state.get(key)
        if result is None:
            continue
        for evidence in result.get("evidence", []):
            if evidence.get("verification_status") != "source_checked":
                continue
            evidence_id = evidence.get("id")
            if not evidence_id:
                continue
            previous = by_id.get(evidence_id)
            if previous is not None and previous != evidence:
                raise ValueError(f"서로 다른 근거가 같은 ID를 사용합니다: {evidence_id}")
            by_id[evidence_id] = evidence
    return by_id


def checked_inline_ids(texts: list[str], valid_ids: set[str]) -> list[str]:
    """본문의 [근거 ID]를 등장 순서대로 추출하고 존재하지 않는 ID를 거부한다."""
    cited = list(dict.fromkeys(
        match.group(1).strip()
        for content in texts
        for match in _INLINE_CITATION.finditer(content)
    ))
    unknown = set(cited) - valid_ids
    if unknown:
        raise ValueError(f"본문에 확인되지 않은 근거 ID가 있습니다: {sorted(unknown)}")
    return cited
