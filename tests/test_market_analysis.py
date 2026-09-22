"""시장 조사 근거의 구조화 평가 변환 테스트."""

import unittest

from kv_cache_eval.features.market.analysis import (
    CitedAssessment,
    GrowthMetric,
    MarketAnalysis,
    SourceCitation,
    SupportFinding,
    materialize_analysis,
)
from kv_cache_eval.features.market.research import SearchHit


class MarketAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.hits = [
            SearchHit(
                "growth",
                "Market report",
                "https://example.com/growth",
                "The AI inference market will grow at a CAGR of 31% from 2025 to 2030.",
                0.9,
                True,
            ),
            SearchHit(
                "adoption",
                "Product docs",
                "https://example.com/adoption",
                "An external pilot deployed this technology in a product test.",
                0.8,
                True,
            ),
            SearchHit(
                "ecosystem",
                "Framework docs",
                "https://example.com/ecosystem",
                "Framework A provides an official integration and API documentation.",
                0.7,
                True,
            ),
        ]

    def analysis(self):
        return MarketAnalysis(
            growth_metric=GrowthMetric(
                market_name="AI inference market",
                cagr_percent=31,
                period_start_year=2025,
                period_end_year=2030,
                base_market_size=100,
                base_year=2024,
                forecast_market_size=300,
                forecast_year=2030,
                currency_and_unit="USD billion",
                citation=SourceCitation(
                    source_url="https://example.com/growth",
                    excerpt="CAGR of 31% from 2025 to 2030",
                ),
            ),
            growth=CitedAssessment(
                judgment="관련 시장이 빠르게 성장함",
                rationale="보고서의 CAGR 기준",
            ),
            adoption_level="external_integration_or_pilot",
            adoption=CitedAssessment(
                judgment="외부 파일럿이 확인됨",
                rationale="제품 문서 기준",
                citations=[
                    SourceCitation(
                        source_url="https://example.com/adoption",
                        excerpt="external pilot deployed this technology",
                    )
                ],
            ),
            supports=[
                SupportFinding(
                    kind="framework_integration",
                    provider="Framework A",
                    citation=SourceCitation(
                        source_url="https://example.com/ecosystem",
                        excerpt="official integration and API documentation",
                    ),
                ),
                SupportFinding(
                    kind="api_or_documentation",
                    provider="Framework A",
                    citation=SourceCitation(
                        source_url="https://example.com/ecosystem",
                        excerpt="official integration and API documentation",
                    ),
                ),
            ],
            ecosystem=CitedAssessment(
                judgment="두 가지 지원 요소가 확인됨",
                rationale="공식 프레임워크 문서 기준",
                citations=[
                    SourceCitation(
                        source_url="https://example.com/ecosystem",
                        excerpt="Framework A provides an official integration",
                    )
                ],
            ),
        )

    def test_scores_and_links_checked_evidence(self):
        research, result = materialize_analysis("KIVI", self.hits, self.analysis())
        self.assertEqual([item["score"] for item in result["evaluations"]], [5, 3, 3])
        self.assertEqual(
            result["evaluations"][0]["market_metric"]["market_name"],
            "AI inference market",
        )
        known_ids = {item["id"] for item in research["evidence"]}
        for evaluation in result["evaluations"]:
            self.assertTrue(evaluation["evidence_ids"])
            self.assertLessEqual(set(evaluation["evidence_ids"]), known_ids)

    def test_rejects_hallucinated_source_url(self):
        analysis = self.analysis()
        analysis.growth_metric.citation.source_url = (
            "https://not-in-search.example/report"
        )
        with self.assertRaisesRegex(ValueError, "검색 결과에 없는 URL"):
            materialize_analysis("KIVI", self.hits, analysis)

    def test_rejects_excerpt_missing_from_source(self):
        analysis = self.analysis()
        analysis.growth_metric.citation.excerpt = "A fabricated CAGR statement"
        with self.assertRaisesRegex(ValueError, "인용문을 확인할 수 없습니다"):
            materialize_analysis("KIVI", self.hits, analysis)

    def test_rejects_cagr_not_present_in_source(self):
        analysis = self.analysis()
        analysis.growth_metric.cagr_percent = 70.9
        with self.assertRaisesRegex(
            ValueError, "원문에서 CAGR 값을 확인할 수 없습니다"
        ):
            materialize_analysis("KIVI", self.hits, analysis)

    def test_metric_values_may_be_outside_short_excerpt_when_present_in_source(self):
        analysis = self.analysis()
        analysis.growth_metric.citation.excerpt = "CAGR of 31%"

        _, result = materialize_analysis("KIVI", self.hits, analysis)

        self.assertEqual(result["evaluations"][0]["score"], 5)

    def test_unverified_snippet_stays_unverified(self):
        unchecked = [self.hits[0]._replace(source_checked=False), *self.hits[1:]]
        research, result = materialize_analysis("KIVI", unchecked, self.analysis())
        self.assertEqual(result["evaluations"][0]["basis_status"], "unverified")
        self.assertEqual(research["evidence"][0]["verification_status"], "unverified")

    def test_ecosystem_links_every_support_source(self):
        analysis = self.analysis()
        analysis.ecosystem.citations = []
        _, result = materialize_analysis("KIVI", self.hits, analysis)
        ecosystem = result["evaluations"][2]
        self.assertEqual(len(ecosystem["evidence_ids"]), 1)
        self.assertEqual(ecosystem["basis_status"], "source_checked")

    def test_author_repository_does_not_raise_ecosystem_score(self):
        author_hit = SearchHit(
            "ecosystem",
            "KIVI author repository",
            "https://github.com/jy-yuan/KIVI?tab=readme-ov-file",
            "The author repository provides setup documentation and active maintenance.",
            0.9,
            True,
        )
        analysis = self.analysis()
        author_citation = SourceCitation(
            source_url=author_hit.url,
            excerpt="setup documentation and active maintenance",
        )
        analysis.supports = [
            SupportFinding(
                kind="api_or_documentation",
                provider="KIVI authors",
                citation=author_citation,
            ),
            SupportFinding(
                kind="active_maintenance",
                provider="KIVI authors",
                citation=author_citation,
            ),
        ]
        analysis.ecosystem.citations = [author_citation]

        research, result = materialize_analysis(
            "KIVI", [*self.hits, author_hit], analysis
        )

        self.assertEqual(result["evaluations"][2]["score"], 1)
        self.assertIn("2건은 점수에서 제외", research["notes"][-1])


if __name__ == "__main__":
    unittest.main()
