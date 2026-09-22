"""시장성 웹 조사 질의와 검색 결과 정규화.

검색 서비스 호출과 평가 판단을 분리해, 네트워크 없이도 질의/중복 제거를
테스트할 수 있게 한다.
"""

from collections.abc import Callable, Iterable, Mapping
import re
import time
from typing import Any, NamedTuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from kv_cache_eval.common.schemas import Technology


class SearchHit(NamedTuple):
    query: str
    title: str
    url: str
    content: str
    score: float | None
    source_checked: bool = False


Search = Callable[[str], Iterable[Mapping[str, Any]]]

COMMON_MARKET_QUERIES = (
    'AI inference market CAGR GPU cloud infrastructure forecast',
    'AI optimized IaaS inference spending forecast',
)

TECHNOLOGY_QUERIES: dict[Technology, tuple[str, ...]] = {
    "KIVI": (
        'KIVI KV cache quantization production adoption',
        'KIVI KV cache framework integration official documentation',
        'KIVI KV cache external implementation GitHub',
    ),
    "InfiniGen": (
        'InfiniGen dynamic KV cache production adoption',
        'InfiniGen KV cache framework integration official documentation',
        'InfiniGen KV cache external implementation GitHub',
    ),
}

_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "ref", "source"}
_MAX_SOURCE_CHARS = 3_000


class TavilyRateLimitError(RuntimeError):
    """세 번 시도한 뒤에도 Tavily 429가 계속될 때의 오류."""


def _is_rate_limit(error: object) -> bool:
    if isinstance(error, TavilyRateLimitError):
        return True
    status = getattr(error, "status_code", None)
    if isinstance(error, Mapping):
        status = error.get("status_code", error.get("status", error.get("code", status)))
    return str(status) == "429" or bool(re.search(r"\b429\b", str(error)))


def _retry_after_seconds(error: object) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(error, Mapping):
        headers = error.get("headers", headers)
    if not isinstance(headers, Mapping):
        return 0
    value = next((item for key, item in headers.items() if str(key).casefold() == "retry-after"), None)
    try:
        return max(0, float(value)) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def canonical_url(url: str) -> str:
    """추적 파라미터와 fragment를 제거해 같은 출처의 중복을 줄인다."""
    value = url.strip()
    if not value:
        raise ValueError("검색 결과 URL이 필요합니다")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"지원하지 않는 검색 결과 URL: {url}")
    query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_") and key.casefold() not in _TRACKING_QUERY_KEYS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, urlencode(query), ""))


def normalize_hits(query: str, raw_hits: Iterable[Mapping[str, Any]]) -> list[SearchHit]:
    """검색 공급자 결과를 검증하고 URL 기준으로 중복 제거한다."""
    hits: list[SearchHit] = []
    seen_urls: set[str] = set()
    for raw in raw_hits:
        url = canonical_url(str(raw.get("url", "")))
        if url in seen_urls:
            continue
        title = str(raw.get("title", "")).strip()
        snippet = str(raw.get("content", "")).strip()
        raw_content = str(raw.get("raw_content") or "").strip()
        content = raw_content or snippet
        if not title or not content:
            raise ValueError(f"제목 또는 본문이 없는 검색 결과: {url}")
        raw_score = raw.get("score")
        score = None if raw_score is None else float(raw_score)
        hits.append(SearchHit(
            query=query,
            title=title,
            url=url,
            content=content[:_MAX_SOURCE_CHARS],
            score=score,
            source_checked=bool(raw_content),
        ))
        seen_urls.add(url)
    return hits


def collect_market_sources(search: Search, technology: Technology) -> list[SearchHit]:
    """공통 시장 수요와 기술별 채택/생태계 질의를 실행한다."""
    collected: list[SearchHit] = []
    seen_urls: set[str] = set()
    for query in (*COMMON_MARKET_QUERIES, *TECHNOLOGY_QUERIES[technology]):
        for hit in normalize_hits(query, search(query)):
            if hit.url not in seen_urls:
                collected.append(hit)
                seen_urls.add(hit.url)
    return collected


def tavily_search(max_results: int = 5) -> Search:
    """한 시장 평가 안에서 중복 질의를 줄이고 Tavily 429만 제한 재시도한다."""
    from langchain_tavily import TavilySearch

    tool = TavilySearch(
        max_results=max_results,
        search_depth="advanced",
        include_raw_content="markdown",
        topic="general",
    )
    cache: dict[str, list[Mapping[str, Any]]] = {}
    last_request_at: float | None = None

    def search(query: str) -> Iterable[Mapping[str, Any]]:
        nonlocal last_request_at
        if query in cache:
            return cache[query]
        for attempt, retry_delay in enumerate((30, 60, 0)):
            if last_request_at is not None:
                remaining = 1 - (time.monotonic() - last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            last_request_at = time.monotonic()
            try:
                response = tool.invoke({"query": query})
                if not isinstance(response, Mapping):
                    raise TypeError("Tavily 검색 결과가 객체 형식이 아닙니다")
                error = response.get("error")
                if error:
                    if _is_rate_limit(error):
                        limited = TavilyRateLimitError("Tavily 429 요청 제한")
                        limited.retry_after = _retry_after_seconds(response) or _retry_after_seconds(error)
                        raise limited
                    raise RuntimeError(f"Tavily 검색 실패: {error}")
                results = response.get("results")
                if not isinstance(results, list):
                    raise TypeError("Tavily 검색 결과에 results 목록이 없습니다")
                cache[query] = results
                return results
            except Exception as exc:
                if not _is_rate_limit(exc):
                    raise
                if attempt == 2:
                    raise TavilyRateLimitError("Tavily 429 요청 제한: 3회 시도 후 중단") from exc
                time.sleep(max(retry_delay, getattr(exc, "retry_after", 0), _retry_after_seconds(exc)))
        raise AssertionError("Tavily 재시도 횟수 오류")

    return search
