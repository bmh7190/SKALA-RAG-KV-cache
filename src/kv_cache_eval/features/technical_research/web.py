"""Tavily 검색 결과의 원문 본문만 일시적인 근거 후보로 만든다."""

from __future__ import annotations

import hashlib
import os
from urllib.parse import urlsplit

from langchain_core.documents import Document


MAX_WEB_RESULTS = 3
MAX_WEB_CHARS = 6000


def _web_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    return value if parsed.scheme in ("http", "https") and parsed.netloc else None


def search_web(query: str, target: str, limit: int = MAX_WEB_RESULTS) -> tuple[list[Document], list[str]]:
    """검색 요약은 버리고 검색 결과의 raw_content만 반환한다."""
    if not os.getenv("TAVILY_API_KEY", "").strip():
        return [], ["웹 검색 미실행: TAVILY_API_KEY 없음"]
    from langchain_tavily import TavilySearch

    count = min(max(limit, 1), MAX_WEB_RESULTS)
    try:
        found = TavilySearch(max_results=count, search_depth="basic", include_answer=False,
                             include_raw_content=True).invoke({"query": query})
    except Exception as error:
        return [], [f"웹 검색 실패({type(error).__name__}); 근거 부재로 판단하지 않음"]
    if not isinstance(found, dict) or not isinstance(found.get("results"), list):
        return [], ["웹 검색 응답 오류; 근거 부재로 판단하지 않음"]
    documents = []
    missing_body = 0
    invalid_result = 0
    for item in found["results"][:count]:
        if not isinstance(item, dict) or not (url := _web_url(item.get("url"))):
            invalid_result += 1
            continue
        body = item.get("raw_content")
        if not isinstance(body, str) or not body.strip():
            missing_body += 1
            continue
        body = body.strip()[:MAX_WEB_CHARS]
        title = str(item.get("title") or url)
        source_id = "web-" + hashlib.sha256(url.encode()).hexdigest()[:16]
        chunk_id = hashlib.sha256((url + body).encode()).hexdigest()[:20]
        documents.append(Document(id=chunk_id, page_content=body, metadata={
            "chunk_id": chunk_id, "source_id": source_id, "source_title": title,
            "source_url": url, "source_kind": "web", "source_role": "web_external",
            "subject_technology": target, "page": None,
        }))
    notes = []
    if invalid_result:
        notes.append(f"웹 검색 결과 URL 오류: {invalid_result}개")
    if missing_body:
        notes.append(f"웹 검색 결과 본문 누락: {missing_body}개; 검색 요약은 근거로 사용하지 않음")
    if not documents:
        notes.append("웹 본문을 확인하지 못함; 검색 요약은 근거로 사용하지 않음")
    return documents, notes
