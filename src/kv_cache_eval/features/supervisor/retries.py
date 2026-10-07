"""실행 오류 재시도와 근거 부족 재조사의 범위·상한을 결정한다."""

from kv_cache_eval.common.tasks import EVALUATION_KEYS
from kv_cache_eval.features.supervisor.catalog import CRITERIA
from kv_cache_eval.features.supervisor.transitions import dispatch, finish
from kv_cache_eval.features.technical_research.prompts import category_for_gap


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
