"""시장성 노드와 공통 근거 게이트 통합 테스트."""

import unittest

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.market.analysis import (
    CitedAssessment,
    GrowthMetric,
    MarketAnalysis,
    SourceCitation,
)
from kv_cache_eval.features.market.node import (
    build_market_node,
    validate_runtime_config,
)
from kv_cache_eval.graph.gates import check_evidence


class MarketNodeTest(unittest.TestCase):
    def test_injected_node_evaluates_both_technologies_without_network(self):
        def search(query):
            slug = str(abs(hash(query)))
            content = f"{query} CAGR 20% from 2025 to 2030"
            return [
                {
                    "title": query,
                    "url": f"https://example.com/{slug}",
                    "content": content,
                    "raw_content": content,
                }
            ]

        def analyse(technology, hits):
            urls = [hit.url for hit in hits]
            return MarketAnalysis(
                growth_metric=GrowthMetric(
                    market_name="AI inference market",
                    cagr_percent=20,
                    period_start_year=2025,
                    period_end_year=2030,
                    citation=SourceCitation(
                        source_url=urls[0],
                        excerpt=hits[0].content,
                    ),
                ),
                growth=CitedAssessment(judgment="성장"),
                adoption_level="public_prototype_only",
                adoption=CitedAssessment(
                    judgment="프로토타입",
                    citations=[
                        SourceCitation(
                            source_url=urls[-2],
                            excerpt=hits[-2].content,
                        )
                    ],
                ),
                supports=[],
                ecosystem=CitedAssessment(
                    judgment="외부 지원 미확인",
                    citations=[
                        SourceCitation(
                            source_url=urls[-1],
                            excerpt=hits[-1].content,
                        )
                    ],
                ),
            )

        update = build_market_node(search, analyse)(new_state())
        self.assertEqual(len(update["market_eval"]["evaluations"]), 6)
        self.assertTrue(update["market_evidence"]["evidence"])

        state = new_state(max_research_rounds=0)
        state.update(update)
        gaps = check_evidence(state)["evidence_gaps"]
        market_gaps = [gap for gap in gaps if gap["criterion"] == "시장성"]
        self.assertEqual(market_gaps, [])


class MarketRuntimeConfigTest(unittest.TestCase):
    def test_reports_all_missing_runtime_settings_together(self):
        with self.assertRaisesRegex(
            RuntimeError, "TAVILY_API_KEY, LLM_PROVIDER, LLM_MODEL"
        ):
            validate_runtime_config({})

    def test_openai_requires_its_api_key(self):
        environment = {
            "TAVILY_API_KEY": "tavily-test",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "test-model",
        }
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            validate_runtime_config(environment)

    def test_returns_normalized_runtime_config(self):
        environment = {
            "TAVILY_API_KEY": "tavily-test",
            "LLM_PROVIDER": " OpenAI ",
            "LLM_MODEL": " test-model ",
            "OPENAI_API_KEY": "openai-test",
        }
        config = validate_runtime_config(environment)
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "test-model")


if __name__ == "__main__":
    unittest.main()
