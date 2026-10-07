"""Canonical evidence collection and citation checks shared by gates and writers."""

import re

EVIDENCE_KEYS = (
    "kivi_evidence",
    "infinigen_evidence",
    "market_evidence",
    "domain_evidence",
)
CITATION = re.compile(r"\[([^\[\]]+)\]")


def collect_evidence(state):
    result = {}
    for key in EVIDENCE_KEYS:
        for item in (state.get(key) or {}).get("evidence", []):
            if item.get("verification_status") != "source_checked":
                continue
            if not item.get("claim", "").strip() or not any(
                item.get("source", {}).get(k) for k in ("document", "url")
            ):
                continue
            normalized = {
                **item,
                "experiment": {
                    k: v
                    for k, v in (item.get("experiment") or {}).items()
                    if v is not None
                },
            }
            if item["id"] in result:
                previous = result[item["id"]]
                comparable = {
                    **previous,
                    "experiment": {
                        k: v
                        for k, v in (previous.get("experiment") or {}).items()
                        if v is not None
                    },
                }
                if comparable != normalized:
                    raise ValueError(f"상충하는 근거 ID: {item['id']}")
            result.setdefault(item["id"], item)
    return result


def validate_citations(texts, evidence, declared=()):
    """Never delete or silently repair an unknown citation."""
    ids = [value.strip() for text in texts for value in CITATION.findall(text)]
    unknown = set([*ids, *declared]) - evidence.keys()
    if unknown:
        raise ValueError(f"확인되지 않은 근거 ID: {sorted(unknown)}")
    return list(dict.fromkeys(ids))


def require_values(state, *keys):
    missing = [key for key in keys if state.get(key) is None]
    if missing:
        raise ValueError(f"선행 결과가 없습니다: {missing}")
    return {key: state[key] for key in keys}
