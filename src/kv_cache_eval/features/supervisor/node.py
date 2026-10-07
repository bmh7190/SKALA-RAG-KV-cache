"""이전 결과 확인 → 조사 → 관점 평가 → 근거 확인 → 보고서·품질·저장 순서로 판단한다."""

from copy import deepcopy

from kv_cache_eval.common.state import State
from kv_cache_eval.common.tasks import EVALUATION_KEYS
from kv_cache_eval.features.report.feedback import issue_sections
from kv_cache_eval.features.supervisor.catalog import CRITERIA
from kv_cache_eval.features.supervisor.evidence_policy import (
    check_evidence,
    research_gaps,
)
from kv_cache_eval.features.supervisor.retries import failed_work, retry_gap
from kv_cache_eval.features.supervisor.transitions import (
    accept_result,
    dispatch,
    finish,
)


def supervise(input_state: State) -> State:
    """입력은 보존하고 한 단계의 배정 또는 종료 결정을 반환한다."""
    state = deepcopy(input_state)
    if state["status"] != "running":
        return state
    decision = _handle_previous_result(state)
    if decision is not None:
        return decision
    completed = set(state["completed_agents"])
    if "technical_research" not in completed:
        return dispatch(
            state, "technical_research", "선정 기술의 원문 근거가 필요합니다"
        )
    # 원문 근거가 0건인 기술을 그대로 평가하면 평가를 두 번 수행하게 된다.
    # 기술별 재조사 1회는 retry_gap의 gap_attempts가 제한한다.
    decision = retry_gap(state, research_gaps(state))
    if decision is not None:
        return decision
    decision = _dispatch_pending_evaluation(state, completed)
    if decision is not None:
        return decision
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
    return _dispatch_report_stage(state, completed)


def _handle_previous_result(state):
    """현재 배정의 결과만 수락한다. 오류·PDF 완료는 즉시 다음 상태를 결정한다."""
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
    return None


def _dispatch_pending_evaluation(state, completed):
    """변경된 근거의 재평가를 우선하고, 호출 수와 미검토 항목으로 순서를 정한다."""
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
    return None


def _dispatch_report_stage(state, completed):
    """초안 → 최신 버전 품질 검사 → 수정 또는 PDF 저장을 선택한다."""
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
        sections = issue_sections(quality["issues"])
        reason = "; ".join(issue["required_action"] for issue in quality["issues"])
        return dispatch(
            state,
            "report",
            reason,
            {"technology": None, "criteria": sections, "reason": reason},
        )
    return dispatch(state, "export_pdf", "현재 버전이 네 품질 기준을 통과했습니다")
