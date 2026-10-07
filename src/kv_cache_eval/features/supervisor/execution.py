"""One worker boundary for validation, change tracking and sanitized error results."""

from copy import deepcopy
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from pydantic import TypeAdapter

from kv_cache_eval.common import schemas
from kv_cache_eval.common.tasks import (
    EVALUATION_KEYS,
    InputBudgetExceeded,
    criteria,
    technologies,
)
from kv_cache_eval.features.domain.results import DomainResponseError
from kv_cache_eval.features.report.node import ReportTooLong
from kv_cache_eval.features.supervisor.catalog import CRITERIA, OUTPUTS

TYPES = {
    **{
        key: schemas.ResearchResult
        for key in (
            "kivi_evidence",
            "infinigen_evidence",
            "market_evidence",
            "domain_evidence",
        )
    },
    **{key: schemas.EvaluationResult for key in EVALUATION_KEYS.values()},
    "synthesis": schemas.Synthesis,
    "report": schemas.ReportDraft,
    "quality_result": schemas.QualityResult,
    "pdf_path": str,
}


class InvalidRequest(ValueError):
    """The supervisor requested a scope the worker cannot handle."""


def worker(agent, function):
    """요청 확인 → 실행 → 결과 검증 → 부족 근거 확인 → 실행 결과 반환."""

    def execute(state, config: RunnableConfig):
        try:
            selected = _validate_request(state, agent)
            update = function(deepcopy(state))
            update = _validate_update(state, agent, update, selected)
            gaps = _find_evidence_gaps(state, agent, update)
            result = {
                "agent": agent,
                "step": state["step_count"],
                "status": "needs_evidence" if gaps else "completed",
                "gaps": gaps,
                "error": None,
                "changed_keys": _changed_keys(state, update),
            }
            return {**update, "last_result": result}
        except Exception as error:
            return {
                "last_result": {
                    "agent": agent,
                    "step": state["step_count"],
                    "status": "failed",
                    "gaps": [],
                    "error": classify_error(error),
                    "changed_keys": [],
                }
            }

    execute.__name__ = agent
    return execute


def _validate_request(state, agent):
    """요청한 기술과 평가 항목이 작업자의 처리 범위 안에 있는지 확인한다."""
    try:
        selected = technologies(state)
        criteria(state, CRITERIA[agent])
    except ValueError as error:
        raise InvalidRequest(str(error)) from error
    return selected


def _validate_update(state, agent, update, selected):
    """허용 필드·필수 결과·스키마·현재 보고서 버전을 확인한다."""
    if not isinstance(update, dict) or set(update) - set(OUTPUTS[agent]):
        raise ValueError(f"{agent}: 허용하지 않은 State 필드 변경")
    if not update:
        raise ValueError(f"{agent}: 결과 누락")
    if agent == "technical_research":
        required = {
            "kivi_evidence" if tech == "KIVI" else "infinigen_evidence"
            for tech in selected
        }
    elif agent in EVALUATION_KEYS:
        required = {EVALUATION_KEYS[agent]}
    else:
        required = set(OUTPUTS[agent])
    if required - update.keys():
        raise ValueError(f"{agent}: 필수 결과 누락")
    update = {
        key: TypeAdapter(TYPES[key]).validate_python(value)
        for key, value in update.items()
    }
    if agent == "quality":
        q = update["quality_result"]
        expected = {
            "groundedness",
            "neutrality",
            "bias_control",
            "perspective_coverage",
        }
        if (
            set(q["checks"]) != expected
            or set(q["reasons"]) != expected
            or q["report_revision"] != state["report_revision"]
        ):
            raise ValueError("현재 보고서의 네 품질 판정이 필요합니다")
        if q["passed"] != (
            all(v == "pass" for v in q["checks"].values()) and not q["issues"]
        ):
            raise ValueError("품질 통과와 개별 판정이 일치하지 않습니다")
    if agent == "export_pdf" and not Path(update["pdf_path"]).is_file():
        raise ValueError("저장된 PDF 파일이 없습니다")
    return update


def _find_evidence_gaps(state, agent, update):
    """갱신 결과에서 추가 근거가 필요한 항목을 찾는다."""
    from kv_cache_eval.graph.gates import assessment_gaps

    merged = {**state, **update}
    gaps = (
        assessment_gaps(merged, agent, scoped=True) if agent in EVALUATION_KEYS else []
    )
    if agent == "technical_research":
        gaps = [
            {
                "technology": tech,
                "criterion": "기술 조사",
                "reason": "검색된 근거 없음",
            }
            for tech in technologies(state)
            if not (
                merged.get("kivi_evidence" if tech == "KIVI" else "infinigen_evidence")
                or {}
            ).get("evidence")
        ]
    return gaps


def _changed_keys(state, update):
    """진단용 notes 변경은 제외하고 실제 결과가 바뀐 필드만 반환한다."""
    changed = []
    for key, value in update.items():
        previous = state.get(key)
        # Notes are diagnostics, not evidence changes.
        old = (
            previous.get("evidence")
            if key.endswith("_evidence") and previous
            else previous
        )
        new = value.get("evidence") if key.endswith("_evidence") else value
        if old != new:
            changed.append(key)
    return changed


def classify_error(error):
    status = getattr(error, "status_code", None)
    name = type(error).__name__.lower()
    if isinstance(error, DomainResponseError):
        code, retry = "response_repair_exhausted", False
    elif isinstance(error, InvalidRequest):
        code, retry = "invalid_request", False
    elif isinstance(error, InputBudgetExceeded):
        code, retry = "input_budget_exceeded", False
    elif isinstance(error, ReportTooLong):
        code, retry = "report_too_long", False
    elif status in (401, 403) or "authentication" in name:
        code, retry = "authentication", False
    elif status == 429 or "ratelimit" in name:
        code, retry = "rate_limit", True
    elif "timeout" in name or isinstance(error, TimeoutError):
        code, retry = "timeout", True
    elif "connection" in name or (status is not None and status >= 500):
        code, retry = "connection", True
    elif isinstance(error, (ValueError, TypeError)):
        code, retry = "invalid_result", False
    else:
        code, retry = "execution_error", False
    # Never include raw provider bodies (which can contain requests/credentials).
    message = (
        str(error)
        if isinstance(error, ReportTooLong)
        else f"{type(error).__name__}: {code}"
    )
    return {"code": code, "message": message, "retryable": retry}
