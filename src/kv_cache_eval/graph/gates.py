"""Evidence sufficiency checks used by the Supervisor, without model calls."""

from kv_cache_eval.common.evidence import collect_evidence
from kv_cache_eval.common.tasks import EVALUATION_KEYS, criteria, technologies
from kv_cache_eval.features.supervisor.catalog import CRITERIA


def validate_input(state):
    if tuple(state["selected_technologies"]) != ("KIVI", "InfiniGen"):
        raise ValueError("현재 그래프는 KIVI와 InfiniGen 비교용입니다")
    if not state["domain_and_criteria"]["domain"].strip():
        raise ValueError("평가 도메인이 필요합니다")
    return {}


def supported(row, evidence):
    ids = row.get("evidence_ids", [])
    return bool(
        (row.get("judgment") is not None or row.get("score") is not None)
        and row.get("basis_status") != "unverified"
        and ids
        and all(
            eid in evidence and evidence[eid]["technology"] == row["technology"]
            for eid in ids
        )
    )


def assessment_gaps(state, agent, *, scoped=False):
    evidence = collect_evidence(state)
    rows = (state.get(EVALUATION_KEYS[agent]) or {}).get("evaluations", [])
    requested = criteria(state, CRITERIA[agent]) if scoped else CRITERIA[agent]
    targets = technologies(state) if scoped else state["selected_technologies"]
    return [
        {
            "technology": technology,
            "criterion": criterion,
            "reason": "판단·출처 근거가 미확인",
        }
        for technology in targets
        for criterion in requested
        if not any(
            row["technology"] == technology
            and row["criterion"] == criterion
            and supported(row, evidence)
            for row in rows
        )
    ]


def check_evidence(state):
    evidence = collect_evidence(state)
    gaps, blocking = [], []
    for technology in state["selected_technologies"]:
        if not any(
            item["technology"] == technology and not eid.startswith("market-")
            for eid, item in evidence.items()
        ):
            gap = {
                "technology": technology,
                "criterion": "기술 조사",
                "reason": "확인된 기술 원문 근거가 없음",
            }
            gaps.append(gap)
            blocking.append(gap)
    for agent, key in EVALUATION_KEYS.items():
        current = assessment_gaps(state, agent)
        gaps.extend(current)
        rows = (state.get(key) or {}).get("evaluations", [])
        for technology in state["selected_technologies"]:
            # A whole missing perspective must never be waved through as a limitation.
            if not any(
                row["technology"] == technology and supported(row, evidence)
                for row in rows
            ):
                blocking.extend(
                    gap for gap in current if gap["technology"] == technology
                )
    ready = not blocking
    return {
        "evidence_gaps": gaps,
        "evidence_decision": {
            "ready": ready,
            "reason": "네 관점의 근거를 확인함. 남은 항목은 한계로 명시함"
            if ready
            else "기술·관점 전체의 근거 부족",
            "blocking_gaps": blocking,
        },
    }
