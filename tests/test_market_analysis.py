"""시장 조사 근거의 구조화 평가 변환 테스트."""

import unittest

from kv_cache_eval.features.market.analysis import (
    CitedAssessment,
    MarketAnalysis,
    SupportFinding,
    materialize_analysis,
)
from kv_cache_eval.features.market.research import SearchHit


class MarketAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.hits = [
            SearchHit("growth", "Market report", "https://example.com/growth", "CAGR 31%", 0.9, True),
            SearchHit("adoption", "Product docs", "https://example.com/adoption", "Pilot", 0.8, True),
            SearchHit("ecosystem", "Framework docs", "https://example.com/ecosystem", "API docs", 0.7, True),
        ]

    def analysis(self):
        return MarketAnalysis(
            cagr_percent=31,
            growth=CitedAssessment(
                judgment="관련 시장이 빠르게 성장함",
                rationale="보고서의 CAGR 기준",
                source_urls=["https://example.com/growth"],
            ),
            adoption_level="external_integration_or_pilot",
            adoption=CitedAssessment(
                judgment="외부 파일럿이 확인됨",
                rationale="제품 문서 기준",
                source_urls=["https://example.com/adoption"],
            ),
            supports=[
                SupportFinding(
                    kind="framework_integration",
                    provider="Framework A",
                    source_url="https://example.com/ecosystem",
                ),
                SupportFinding(
                    kind="api_or_documentation",
                    provider="Framework A",
                    source_url="https://example.com/ecosystem",
                ),
            ],
            ecosystem=CitedAssessment(
                judgment="두 가지 지원 요소가 확인됨",
                rationale="공식 프레임워크 문서 기준",
                source_urls=["https://example.com/ecosystem"],
            ),
        )

    def test_scores_and_links_checked_evidence(self):
        research, result = materialize_analysis("KIVI", self.hits, self.analysis())
        self.assertEqual([item["score"] for item in result["evaluations"]], [5, 3, 3])
        known_ids = {item["id"] for item in research["evidence"]}
        for evaluation in result["evaluations"]:
            self.assertTrue(evaluation["evidence_ids"])
            self.assertLessEqual(set(evaluation["evidence_ids"]), known_ids)

    def test_rejects_hallucinated_source_url(self):
        analysis = self.analysis()
        analysis.growth.source_urls = ["https://not-in-search.example/report"]
        with self.assertRaisesRegex(ValueError, "검색 결과에 없는 URL"):
            materialize_analysis("KIVI", self.hits, analysis)

    def test_unverified_snippet_stays_unverified(self):
        unchecked = [self.hits[0]._replace(source_checked=False), *self.hits[1:]]
        research, result = materialize_analysis("KIVI", unchecked, self.analysis())
        self.assertEqual(result["evaluations"][0]["basis_status"], "unverified")
        self.assertEqual(research["evidence"][0]["verification_status"], "unverified")

    def test_ecosystem_links_every_support_source(self):
        analysis = self.analysis()
        analysis.ecosystem.source_urls = []
        _, result = materialize_analysis("KIVI", self.hits, analysis)
        ecosystem = result["evaluations"][2]
        self.assertEqual(len(ecosystem["evidence_ids"]), 1)
        self.assertEqual(ecosystem["basis_status"], "source_checked")


if __name__ == "__main__":
    unittest.main()
