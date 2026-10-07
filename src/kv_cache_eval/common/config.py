"""공통 환경 설정. import만으로 .env나 모델을 읽고 실행하지 않는다."""

import logging
import os
from collections import ChainMap
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

MAX_SELECTED_SOURCE_PAGES = 200
EMBEDDING_MODEL = "BAAI/bge-m3"


def load_environment(path: str | Path = ".env") -> bool:
    """선택한 .env를 명시적으로 읽는다. 이미 설정된 셸 값은 유지한다."""
    return load_dotenv(dotenv_path=path, override=False)


def get_embedding_model() -> str:
    """환경변수에 지정된 모델 또는 사용자 지정 기본 모델명을 돌려준다."""
    return os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL


def model_settings(
    *,
    role: str = "generator",
    timeout: float = 120,
    file_values: Mapping[str, str | None] | None = None,
    provider_names: tuple[str, ...] = ("LLM_PROVIDER",),
) -> dict:
    """셸 우선순위로 모델 설정을 검증한다. 환경변수는 변경하지 않는다."""
    file_values = file_values or {}
    values = ChainMap(os.environ, file_values)

    def setting(name):
        return (values.get(name) or "").strip()

    # 별칭도 하나의 계층에서 해석해야 파일 값이 셸 설정과 충돌하지 않는다.
    provider_values = (
        os.environ
        if any(name in os.environ for name in provider_names)
        else file_values
    )
    providers = {
        (provider_values.get(name) or "").strip().lower()
        for name in provider_names
        if name in provider_values
    }
    if len(providers) > 1:
        raise ValueError(
            f"{'와 '.join(provider_names)} 값이 서로 다릅니다. 같은 값으로 맞춰 주세요."
        )
    if providers != {"openai"}:
        names = " 또는 ".join(f"{name}=openai" for name in provider_names)
        raise ValueError(f"{names} 설정이 필요합니다")
    model = (setting("JUDGE_MODEL") if role == "judge" else "") or setting("LLM_MODEL")
    api_key = setting("OPENAI_API_KEY")
    if not model or not api_key:
        raise ValueError("LLM_MODEL과 OPENAI_API_KEY 설정이 필요합니다")
    file_model = (file_values.get("LLM_MODEL") or "").strip()
    if "LLM_MODEL" in os.environ and file_model and setting("LLM_MODEL") != file_model:
        logging.getLogger(__name__).warning(
            "실행 환경의 LLM_MODEL이 설정 파일과 달라 실행 환경 값을 사용합니다. 모델 설정을 확인하세요."
        )
    return dict(model=model, api_key=api_key, timeout=timeout, max_retries=0)
