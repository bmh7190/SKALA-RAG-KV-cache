"""시장성 웹 검색과 출처 정규화 테스트."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kv_cache_eval.features.market.research import (
    COMMON_MARKET_QUERIES,
    TECHNOLOGY_QUERIES,
    TavilyRateLimitError,
    canonical_url,
    collect_market_sources,
    tavily_search,
)


class MarketResearchTest(unittest.TestCase):
    def test_429_dict_retries_with_backoff_then_caches_and_paces(self):
        responses = [
            {"error": {"status_code": 429, "message": "limited"}},
            {"error": "Error 429: excessive requests"},
            {"results": [{"title": "ok", "url": "https://example.com/ok", "raw_content": "ok"}]},
            {"results": []},
        ]
        calls, sleeps, clock = [], [], [0.0]

        class FakeTool:
            def __init__(self, **kwargs):
                pass

            def invoke(self, payload):
                calls.append(payload["query"])
                return responses.pop(0)

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        with patch.dict("sys.modules", {"langchain_tavily": SimpleNamespace(TavilySearch=FakeTool)}), \
             patch("kv_cache_eval.features.market.research.time.monotonic", side_effect=lambda: clock[0]), \
             patch("kv_cache_eval.features.market.research.time.sleep", side_effect=sleep):
            search = tavily_search()
            first = search("shared query")
            self.assertIs(search("shared query"), first)
            self.assertEqual(search("another query"), [])
        self.assertEqual(calls, ["shared query"] * 3 + ["another query"])
        self.assertEqual(sleeps, [30, 60, 1])

    def test_429_exception_retries_only_three_times(self):
        class Rate429(RuntimeError):
            status_code = 429

        calls = []

        class FakeTool:
            def __init__(self, **kwargs):
                pass

            def invoke(self, payload):
                calls.append(payload["query"])
                raise Rate429("limited")

        with patch.dict("sys.modules", {"langchain_tavily": SimpleNamespace(TavilySearch=FakeTool)}), \
             patch("kv_cache_eval.features.market.research.time.sleep") as sleep:
            with self.assertRaises(TavilyRateLimitError):
                tavily_search()("query")
        self.assertEqual(len(calls), 3)
        self.assertIn(30, [call.args[0] for call in sleep.call_args_list])
        self.assertIn(60, [call.args[0] for call in sleep.call_args_list])

    def test_retry_after_header_extends_wait_when_available(self):
        class Rate429(RuntimeError):
            status_code = 429
            response = SimpleNamespace(headers={"Retry-After": "45"})

        responses = [Rate429("limited"), {"results": []}]
        clock, sleeps = [0.0], []

        class FakeTool:
            def __init__(self, **kwargs):
                pass

            def invoke(self, payload):
                response = responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        with patch.dict("sys.modules", {"langchain_tavily": SimpleNamespace(TavilySearch=FakeTool)}), \
             patch("kv_cache_eval.features.market.research.time.monotonic", side_effect=lambda: clock[0]), \
             patch("kv_cache_eval.features.market.research.time.sleep", side_effect=sleep):
            self.assertEqual(tavily_search()("query"), [])
        self.assertEqual(sleeps, [45])

    def test_non429_errors_are_not_retried(self):
        for error in ({"status_code": 401}, {"status_code": 432}, RuntimeError("server failed")):
            with self.subTest(error=error):
                calls = []

                class FakeTool:
                    def __init__(self, **kwargs):
                        pass

                    def invoke(self, payload):
                        calls.append(payload["query"])
                        if isinstance(error, Exception):
                            raise error
                        return {"error": error}

                with patch.dict("sys.modules", {"langchain_tavily": SimpleNamespace(TavilySearch=FakeTool)}), \
                     patch("kv_cache_eval.features.market.research.time.sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        tavily_search()("query")
                self.assertEqual(calls, ["query"])
                sleep.assert_not_called()

    def test_canonical_url_removes_tracking_but_keeps_meaningful_query(self):
        url = "HTTPS://Example.COM/report/?year=2026&utm_source=newsletter#summary"
        self.assertEqual(canonical_url(url), "https://example.com/report?year=2026")

    def test_collect_sources_runs_queries_and_deduplicates(self):
        called: list[str] = []

        def search(query):
            called.append(query)
            return [{
                "title": "Official source",
                "url": "https://example.com/source?utm_source=duplicate",
                "content": "Verified source content",
                "score": 0.9,
            }]

        hits = collect_market_sources(search, "KIVI")
        self.assertEqual(called, [*COMMON_MARKET_QUERIES, *TECHNOLOGY_QUERIES["KIVI"]])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].url, "https://example.com/source")
        self.assertFalse(hits[0].source_checked)

    def test_raw_content_marks_source_as_checked(self):
        hits = collect_market_sources(
            lambda query: [{
                "title": "Official source",
                "url": "https://example.com/source",
                "content": "Search snippet",
                "raw_content": "Original page body",
            }],
            "KIVI",
        )
        self.assertTrue(hits[0].source_checked)
        self.assertEqual(hits[0].content, "Original page body")


if __name__ == "__main__":
    unittest.main()
