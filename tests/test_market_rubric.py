"""시장성 평가표의 경계값과 중복 처리 테스트."""

import unittest

from kv_cache_eval.features.market.rubric import (
    EcosystemSupport,
    independent_supports,
    score_commercial_adoption,
    score_ecosystem_support,
    score_market_growth,
)


class MarketRubricTest(unittest.TestCase):
    def test_market_growth_boundaries(self):
        cases = (
            (30.0, 5), (29.999, 4), (20.0, 4), (19.999, 3),
            (10.0, 3), (9.999, 2), (0.001, 2), (0.0, 1), (-1.0, 1),
        )
        for cagr, expected in cases:
            with self.subTest(cagr=cagr):
                self.assertEqual(score_market_growth(cagr), expected)

    def test_market_growth_rejects_non_finite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                score_market_growth(value)

    def test_commercial_adoption_levels(self):
        self.assertEqual(score_commercial_adoption("multiple_production"), 5)
        self.assertEqual(score_commercial_adoption("single_production"), 4)
        self.assertEqual(score_commercial_adoption("external_integration_or_pilot"), 3)
        self.assertEqual(score_commercial_adoption("public_prototype_only"), 2)
        self.assertEqual(score_commercial_adoption("research_only"), 1)

    def test_ecosystem_support_deduplicates_repeated_reporting(self):
        supports = [
            EcosystemSupport("framework_integration", "Hugging Face"),
            EcosystemSupport("framework_integration", " hugging   face "),
            EcosystemSupport("api_or_documentation", "Hugging Face"),
            EcosystemSupport("external_implementation", "Example Project"),
        ]
        self.assertEqual(len(independent_supports(supports)), 3)
        self.assertEqual(score_ecosystem_support(supports), 4)

    def test_ecosystem_support_score_boundaries(self):
        supports = [
            EcosystemSupport("framework_integration", "A"),
            EcosystemSupport("external_implementation", "B"),
            EcosystemSupport("api_or_documentation", "C"),
            EcosystemSupport("active_maintenance", "D"),
        ]
        expected_by_count = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        for count, expected in expected_by_count.items():
            with self.subTest(count=count):
                self.assertEqual(score_ecosystem_support(supports[:count]), expected)


if __name__ == "__main__":
    unittest.main()
