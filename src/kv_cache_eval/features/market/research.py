"""시장성 웹 조사 질의와 검색 결과 정규화.

검색 서비스 호출과 평가 판단을 분리해, 네트워크 없이도 질의/중복 제거를
테스트할 수 있게 한다.
"""

from collections.abc import Callable, Iterable, Mapping
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
    """환경변수의 TAVILY_API_KEY를 사용하는 실제 검색 함수를 만든다."""
    from langchain_tavily import TavilySearch

    tool = TavilySearch(
        max_results=max_results,
        search_depth="advanced",
        include_raw_content="markdown",
        topic="general",
    )

    def search(query: str) -> Iterable[Mapping[str, Any]]:
        response = tool.invoke({"query": query})
        if not isinstance(response, Mapping):
            raise TypeError("Tavily 검색 결과가 객체 형식이 아닙니다")
        error = response.get("error")
        if error:
            raise RuntimeError(f"Tavily 검색 실패: {error}")
        results = response.get("results")
        if not isinstance(results, list):
            raise TypeError("Tavily 검색 결과에 results 목록이 없습니다")
        return results

    return search
