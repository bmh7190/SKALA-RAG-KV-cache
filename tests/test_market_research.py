"""시장성 웹 검색과 출처 정규화 테스트."""

import unittest

from kv_cache_eval.features.market.research import (
    COMMON_MARKET_QUERIES,
    TECHNOLOGY_QUERIES,
    canonical_url,
    collect_market_sources,
)


class MarketResearchTest(unittest.TestCase):
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
