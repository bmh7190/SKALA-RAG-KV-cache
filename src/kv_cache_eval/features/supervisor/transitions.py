"""Supervisor가 소유하는 배정·종료·결과 무효화 규칙. 전달된 작업용 State를 갱신한다."""

from kv_cache_eval.common.tasks import EVALUATION_KEYS
from kv_cache_eval.features.supervisor.catalog import WRITING


def finish(state, status, reason):
    state.update(
        status=status,
        termination_reason=reason,
        next_agent=None,
        retry_request=None,
        decision_reason=reason,
    )
    return state


def dispatch(state, agent, reason, request=None):
    if (
        state["step_count"] >= state["max_steps"]
        or state["agent_calls"].get(agent, 0) >= state["max_agent_calls"]
    ):
        return finish(state, "incomplete", f"{agent}: 호출 상한 도달. {reason}")
    if agent == "report" and state["agent_calls"].get("report", 0):
        if state["report_revision"] >= state["max_report_revisions"]:
            return finish(state, "incomplete", "보고서 수정 상한에 도달했습니다")
        state["report_revision"] += 1
    state["step_count"] += 1
    state["agent_calls"][agent] = state["agent_calls"].get(agent, 0) + 1
    state.update(next_agent=agent, retry_request=request, decision_reason=reason)
    state["completed_agents"] = [
        name for name in state["completed_agents"] if name != agent
    ]
    if agent == "report":
        state["completed_agents"] = [
            name
            for name in state["completed_agents"]
            if name not in ("quality", "export_pdf")
        ]
        state["pdf_path"] = None
    return state


def invalidate(state, names):
    state["completed_agents"] = [
        name for name in state["completed_agents"] if name not in names
    ]
    for name in names:
        if name in WRITING:
            state[
                {"quality": "quality_result", "export_pdf": "pdf_path"}.get(name, name)
            ] = None
    if any(name in EVALUATION_KEYS for name in names):
        state["evidence_decision"] = None


def accept_result(state, result):
    agent = result["agent"]
    state["last_error"] = None
    changed = set(result.get("changed_keys", []))
    if changed & {"kivi_evidence", "infinigen_evidence"}:
        market_was_complete = "market" in state["completed_agents"]
        invalidate(state, [*EVALUATION_KEYS, *WRITING])
        target = (state.get("retry_request") or {}).get("technology")
        for name in ("maturity", "stakeholders", "domain"):
            previous = state["pending_work"].get(name)
            technology = (
                target if previous is None or previous["technology"] == target else None
            )
            state["pending_work"][name] = {
                "technology": technology,
                "criteria": [],
                "reason": "기술 근거 변경 반영",
            }
        # Market has its own research; its valid result does not depend on paper retrieval.
        if market_was_complete:
            state["completed_agents"].append("market")
    elif changed & set(EVALUATION_KEYS.values()):
        invalidate(state, WRITING)
    elif agent == "synthesis":
        invalidate(state, ("report", "quality", "export_pdf"))
    elif agent == "report":
        invalidate(state, ("quality", "export_pdf"))
    state["completed_agents"] = list(dict.fromkeys([*state["completed_agents"], agent]))
    state["pending_work"].pop(agent, None)
