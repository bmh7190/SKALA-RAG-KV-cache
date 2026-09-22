"""설정 파일을 실행 위치 또는 명시 경로에서 찾아 LLM에 연결한다."""

import math
import os
import logging
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar


class DomainConfigurationError(RuntimeError):
    """사용자에게 표시할 설정 오류. 비밀 값은 포함하지 않는다."""


def find_env_file():
    explicit = os.getenv("DOMAIN_ENV_FILE", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise DomainConfigurationError("DOMAIN_ENV_FILE로 지정한 설정 파일이 없습니다")
        return path
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        path = directory / ".env"
        if path.is_file():
            return path
    return None


_evaluation_config = ContextVar("domain_evaluation_config", default=None)


@contextmanager
def evaluation_settings():
    """한 평가 안에서는 설정을 고정하고 다음 평가에서는 다시 읽는다."""
    token = _evaluation_config.set({})
    try:
        yield
    finally:
        _evaluation_config.reset(token)


def _load_settings(dotenv_values):
    # 설정을 프로세스 환경에 써 넣지 않아 실행 사이에 값이 섞이지 않게 한다.
    try:
        env_file = find_env_file()
        file_values = dotenv_values(env_file) if env_file is not None else {}
    except (OSError, UnicodeError, ValueError):
        raise DomainConfigurationError(
            "설정 파일을 읽을 수 없습니다. 파일 경로·읽기 권한·UTF-8 인코딩을 확인하세요."
        ) from None
    def setting(name):
        return (os.environ.get(name, file_values.get(name, "")) or "").strip()

    # 두 이름을 지원하되, 같은 설정 계층에서 서로 다른 값이면 명시적으로 알린다.
    provider_names = ("LLM_PROVIDER", "LM_PROVIDER")
    provider_values = os.environ if any(name in os.environ for name in provider_names) else file_values
    providers = {(provider_values.get(name) or "").strip().lower()
                 for name in provider_names if name in provider_values}
    if len(providers) > 1:
        raise DomainConfigurationError("LLM_PROVIDER와 LM_PROVIDER 값이 서로 다릅니다. 같은 값으로 맞춰 주세요.")
    if providers != {"openai"}:
        raise DomainConfigurationError("LLM_PROVIDER=openai 또는 LM_PROVIDER=openai 설정이 필요합니다")
    settings = {name: setting(name) for name in ("LLM_MODEL", "OPENAI_API_KEY")}
    file_model = (file_values.get("LLM_MODEL") or "").strip()
    if "LLM_MODEL" in os.environ and file_model and settings["LLM_MODEL"] != file_model:
        logging.getLogger(__name__).warning(
            "실행 환경의 LLM_MODEL이 설정 파일과 달라 실행 환경 값을 사용합니다. 모델 설정을 확인하세요.",
        )
    try:
        timeout = float(setting("DOMAIN_LLM_TIMEOUT_SECONDS") or "45")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except ValueError:
        raise DomainConfigurationError("DOMAIN_LLM_TIMEOUT_SECONDS는 양의 유한한 숫자여야 합니다") from None
    model_name = settings["LLM_MODEL"]
    if not model_name or not settings["OPENAI_API_KEY"]:
        raise DomainConfigurationError("LLM_MODEL과 OPENAI_API_KEY 설정이 필요합니다")
    return dict(model=model_name, api_key=settings["OPENAI_API_KEY"],
                timeout=timeout, max_retries=0)


def invoke_structured(messages, schema):
    try:
        from dotenv import dotenv_values
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise DomainConfigurationError(
            "LLM 패키지가 필요합니다: uv sync --locked --extra llm-openai"
        ) from exc
    cache = _evaluation_config.get()
    if cache is None:
        settings = _load_settings(dotenv_values)
    else:
        if "settings" not in cache:
            cache["settings"] = _load_settings(dotenv_values)
        settings = cache["settings"]
    try:
        model = ChatOpenAI(**settings)
        return model.with_structured_output(schema, method="json_schema", strict=True).invoke(messages)
    except Exception as exc:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = getattr(exc, "code", None) or (error.get("code") if isinstance(error, dict) else None)
        if getattr(exc, "status_code", None) == 429 and code in ("insufficient_quota", "billing_hard_limit_reached"):
            raise DomainConfigurationError("LLM 사용량 또는 결제 한도 확인이 필요합니다. 재검색으로 해결되지 않습니다.") from None
        if getattr(exc, "status_code", None) in (400, 401, 403, 404):
            raise DomainConfigurationError(
                "LLM 요청 설정·모델·인증·권한을 확인해야 합니다. 논문 재검색으로 해결되지 않습니다."
            ) from None
        raise
