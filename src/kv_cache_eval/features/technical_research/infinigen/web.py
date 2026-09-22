"""Tavily 검색으로 URL을 찾고 Extract 본문만 일시적인 근거 후보로 만든다."""

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
    """검색 요약은 버리고 검색 URL에 대응하는 Extract 본문만 반환한다."""
    if not os.getenv("TAVILY_API_KEY", "").strip():
        return [], ["웹 검색 미실행: TAVILY_API_KEY 없음"]
    from langchain_tavily import TavilyExtract, TavilySearch

    count = min(max(limit, 1), MAX_WEB_RESULTS)
    try:
        found = TavilySearch(max_results=count, search_depth="basic", include_answer=False,
                             include_raw_content=False).invoke({"query": query})
    except Exception as error:
        return [], [f"웹 검색 실패({type(error).__name__}); 근거 부재로 판단하지 않음"]
    if not isinstance(found, dict):
        return [], ["웹 검색 응답 오류; 근거 부재로 판단하지 않음"]
    candidates = {}
    for item in found.get("results", [])[:count]:
        if isinstance(item, dict) and (url := _web_url(item.get("url"))):
            candidates[url.rstrip("/")] = (url, str(item.get("title") or url))
    if not candidates:
        return [], ["웹 검색에서 유효한 URL을 찾지 못함; 근거 부재로 판단하지 않음"]

    try:
        extracted = TavilyExtract(extract_depth="basic", format="text", chunks_per_source=1).invoke({
            "urls": [value[0] for value in candidates.values()], "query": query,
        })
    except Exception as error:
        return [], [f"웹 본문 추출 실패({type(error).__name__}); 근거 부재로 판단하지 않음"]
    if not isinstance(extracted, dict):
        return [], ["웹 본문 추출 응답 오류; 근거 부재로 판단하지 않음"]

    documents = []
    for item in extracted.get("results", []):
        if not isinstance(item, dict) or not (url := _web_url(item.get("url"))):
            continue
        candidate = candidates.get(url.rstrip("/"))
        if candidate is None:
            continue  # 검색 도구가 제시하지 않은 URL은 사용하지 않는다.
        body = item.get("raw_content") or item.get("content")
        if not isinstance(body, str) or not body.strip():
            continue
        body = body.strip()[:MAX_WEB_CHARS]
        source_url, title = candidate
        source_id = "web-" + hashlib.sha256(source_url.encode()).hexdigest()[:16]
        chunk_id = hashlib.sha256((source_url + body).encode()).hexdigest()[:20]
        documents.append(Document(id=chunk_id, page_content=body, metadata={
            "chunk_id": chunk_id, "source_id": source_id, "source_title": title,
            "source_url": source_url, "source_kind": "web", "source_role": "web_external",
            "subject_technology": target, "page": None,
        }))
    notes = []
    if extracted.get("failed_results"):
        notes.append(f"웹 본문 일부 추출 실패: {len(extracted['failed_results'])}개 URL")
    if not documents:
        notes.append("웹 본문을 확인하지 못함; 검색 요약은 근거로 사용하지 않음")
    return documents, notes
