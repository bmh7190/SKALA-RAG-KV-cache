"""작업자 실패의 공통 계약: 오류 코드, 재시도 여부, 메시지 공개 여부.

기능(features)은 아래 기반 예외를 상속해 자신의 실패를 표현한다.
Supervisor 실행 경계는 기능별 예외 클래스를 import하지 않고
`classify_error()`로 State에 기록할 `AgentError`를 만든다.
"""

from kv_cache_eval.common.schemas import AgentError

# Supervisor의 재시도·종료 정책이 참조하는 오류 코드
INVALID_REQUEST = "invalid_request"
INVALID_RESULT = "invalid_result"
INPUT_BUDGET_EXCEEDED = "input_budget_exceeded"
RESPONSE_REPAIR_EXHAUSTED = "response_repair_exhausted"
REPORT_TOO_LONG = "report_too_long"
AUTHENTICATION = "authentication"
RATE_LIMIT = "rate_limit"
TIMEOUT = "timeout"
CONNECTION = "connection"
EXECUTION_ERROR = "execution_error"


class AgentFailure(ValueError):
    """Supervisor가 코드로 해석할 수 있는 작업자 실패.

    하위 클래스가 `code`·`retryable`·`expose_message`를 지정한다.
    ValueError를 상속해 기존 `except ValueError` 처리와 호환된다.
    """

    code = INVALID_RESULT
    retryable = False
    # True면 메시지를 State에 그대로 남긴다. 코드가 직접 만든 안전한 문장에만 사용한다.
    expose_message = False


class InvalidRequest(AgentFailure):
    """Supervisor가 작업자가 처리할 수 없는 범위를 요청했다."""

    code = INVALID_REQUEST


class InputBudgetExceeded(AgentFailure):
    """모델 입력 한도를 넘어 같은 요청을 반복해도 해결되지 않는다."""

    code = INPUT_BUDGET_EXCEEDED


class ResponseRepairExhausted(AgentFailure):
    """제한된 재작성 후에도 검증 가능한 모델 응답을 만들지 못했다."""

    code = RESPONSE_REPAIR_EXHAUSTED


def classify_error(error: BaseException) -> AgentError:
    """예외를 State에 남길 수 있는 `AgentError`로 바꾼다."""
    if isinstance(error, AgentFailure):
        code, retryable = error.code, error.retryable
    else:
        code, retryable = _classify_external(error)
    # 외부 예외 원문에는 요청 본문·자격 증명이 섞일 수 있어 기록하지 않는다.
    message = (
        str(error)
        if getattr(error, "expose_message", False)
        else f"{type(error).__name__}: {code}"
    )
    return {"code": code, "message": message, "retryable": retryable}


def _classify_external(error):
    """모델·검색 제공자 등 외부 예외를 HTTP 상태와 클래스 이름으로 분류한다."""
    status = getattr(error, "status_code", None)
    name = type(error).__name__.lower()
    if status in (401, 403) or "authentication" in name:
        return AUTHENTICATION, False
    if status == 429 or "ratelimit" in name:
        return RATE_LIMIT, True
    if "timeout" in name or isinstance(error, TimeoutError):
        return TIMEOUT, True
    if "connection" in name or (status is not None and status >= 500):
        return CONNECTION, True
    if isinstance(error, (ValueError, TypeError)):
        return INVALID_RESULT, False
    return EXECUTION_ERROR, False
