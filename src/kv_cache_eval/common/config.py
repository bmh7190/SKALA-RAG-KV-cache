"""공통 환경 설정. import만으로 .env나 모델을 읽고 실행하지 않는다."""

import os
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
