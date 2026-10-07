"""Choose the next eligible work from current results, gaps and quality feedback."""

from copy import deepcopy

from kv_cache_eval.common.tasks import EVALUATION_KEYS
from kv_cache_eval.features.supervisor.catalog import CRITERIA, WRITING
from kv_cache_eval.features.technical_research.prompts import category_for_gap
from kv_cache_eval.graph.gates import check_evidence


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


def failed_work(state, result):
    agent, error = result["agent"], result["error"]
    state["last_error"] = error
    if error["code"] == "report_too_long":
        return dispatch(
            state,
            "report",
            error["message"],
            {"technology": None, "criteria": [], "reason": error["message"]},
        )
    if error["retryable"]:
        if (
            state["agent_calls"].get(agent, 0) >= state["max_agent_calls"]
            or state["step_count"] >= state["max_steps"]
        ):
            return finish(
                state, "failed", f"{agent}: 실행 오류 재시도 소진 ({error['code']})"
            )
        return dispatch(
            state,
            agent,
            f"{error['code']}: 일시적 오류 재시도",
            state.get("retry_request"),
        )
    if error["code"] == "invalid_result" and agent in (
        "report",
        "synthesis",
        "quality",
    ):
        return dispatch(
            state,
            agent,
            "출력 형식·인용 검증 실패. 허용된 스키마와 근거 ID로 재작성한다.",
            {
                "technology": None,
                "criteria": [],
                "reason": "출력 형식·인용 검증 실패. 허용된 스키마와 근거 ID로 재작성한다.",
            },
        )
    return finish(state, "failed", f"{agent}: {error['message']}")


def retry_gap(state, gaps):
    """At most one focused search for each technology/perspective before reassessing."""
    for gap in gaps:
        technology, criterion = gap["technology"], gap["criterion"]
        owner = next(
            (name for name in EVALUATION_KEYS if criterion in CRITERIA[name]),
            "technical_research",
        )
        key = f"{technology}:{owner}"
        if state["gap_attempts"].get(key, 0) >= 1:
            continue
        agent = "market" if owner == "market" else "technical_research"
        if state["agent_calls"].get(agent, 0) >= state["max_agent_calls"]:
            continue
        related = [
            item
            for item in gaps
            if item["technology"] == technology
            and item["criterion"] in CRITERIA.get(owner, ())
        ]
        if owner == "market":
            selected = list(dict.fromkeys(item["criterion"] for item in related))
        elif owner == "maturity":
            selected = [
                "principle",
                "mechanism",
                "experiment_conditions",
                "performance_results",
            ]
        else:
            selected = list(
                dict.fromkeys(
                    category_for_gap(item["criterion"], item["reason"])
                    for item in related
                )
            ) or ["experiment_conditions"]
        state["gap_attempts"][key] = state["gap_attempts"].get(key, 0) + 1
        # Re-evaluate the requesting perspective even if no new evidence was found.
        if owner in EVALUATION_KEYS and owner != "market":
            state["pending_work"][owner] = {
                "technology": technology,
                "criteria": list(dict.fromkeys(item["criterion"] for item in related)),
                "reason": "추가 조사 결과를 반영해 미확인 항목 재검토",
            }
        return dispatch(
            state,
            agent,
            gap["reason"],
            {"technology": technology, "criteria": selected, "reason": gap["reason"]},
        )
    return None


def supervise(input_state):
    state = deepcopy(input_state)
    if state["status"] != "running":
        return state
    result = state.get("last_result")
    state["last_result"] = None
    if result:
        if (
            result["agent"] != state["next_agent"]
            or result["step"] != state["step_count"]
        ):
            return finish(state, "failed", "현재 배정과 작업 결과가 일치하지 않습니다")
        if result["status"] == "failed":
            return failed_work(state, result)
        accept_result(state, result)
        if result["agent"] == "export_pdf":
            return finish(
                state, "completed", "최신 보고서 품질과 PDF 저장을 확인했습니다"
            )
    completed = set(state["completed_agents"])
    if "technical_research" not in completed:
        return dispatch(
            state, "technical_research", "선정 기술의 원문 근거가 필요합니다"
        )
    pending = [
        name
        for name in EVALUATION_KEYS
        if name not in completed or name in state["pending_work"]
    ]
    if pending:
        # Priority depends on stale results, work already attempted and uncovered items.
        agent = min(
            pending,
            key=lambda name: (
                name not in state["pending_work"],
                state["agent_calls"].get(name, 0),
                -len(CRITERIA[name]),
                name,
            ),
        )
        request = state["pending_work"].get(agent)
        return dispatch(
            state,
            agent,
            (request or {}).get("reason", "아직 검토하지 않은 관점 평가"),
            request,
        )
    state.update(check_evidence(state))
    if "synthesis" not in completed:
        retry = retry_gap(
            state, state["evidence_decision"]["blocking_gaps"] or state["evidence_gaps"]
        )
        if retry is not None:
            return retry
        if not state["evidence_decision"]["ready"]:
            return finish(
                state, "incomplete", "추가 조사 후에도 필수 관점의 근거가 부족합니다"
            )
        return dispatch(state, "synthesis", state["evidence_decision"]["reason"])
    if "report" not in completed:
        return dispatch(
            state, "report", "근거 검토와 종합이 완료되어 초안을 작성합니다"
        )
    if "quality" not in completed:
        return dispatch(
            state, "quality", "현재 초안의 인용·중립성·편향·관점 커버리지를 검사합니다"
        )
    quality = state["quality_result"]
    if quality is None or quality["report_revision"] != state["report_revision"]:
        return dispatch(
            state, "quality", "현재 보고서 버전에 대한 품질 판정이 필요합니다"
        )
    if not quality["passed"]:
        sections = list(
            dict.fromkeys(
                issue["section"]
                for issue in quality["issues"]
                if issue["section"] in CRITERIA["report"]
            )
        )
        reason = "; ".join(issue["required_action"] for issue in quality["issues"])
        return dispatch(
            state,
            "report",
            reason,
            {"technology": None, "criteria": sections, "reason": reason},
        )
    return dispatch(state, "export_pdf", "현재 버전이 네 품질 기준을 통과했습니다")
