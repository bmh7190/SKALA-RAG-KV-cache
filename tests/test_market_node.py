"""시장성 노드와 공통 근거 게이트 통합 테스트."""

import unittest

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.market.analysis import CitedAssessment, MarketAnalysis
from kv_cache_eval.features.market.node import build_market_node, validate_runtime_config
from kv_cache_eval.graph.gates import check_evidence


class MarketNodeTest(unittest.TestCase):
    def test_injected_node_evaluates_both_technologies_without_network(self):
        def search(query):
            slug = str(abs(hash(query)))
            return [{
                "title": query,
                "url": f"https://example.com/{slug}",
                "content": query,
                "raw_content": query,
            }]

        def analyse(technology, hits):
            urls = [hit.url for hit in hits]
            return MarketAnalysis(
                cagr_percent=20,
                growth=CitedAssessment(judgment="성장", source_urls=[urls[0]]),
                adoption_level="public_prototype_only",
                adoption=CitedAssessment(judgment="프로토타입", source_urls=[urls[-2]]),
                supports=[],
                ecosystem=CitedAssessment(judgment="외부 지원 미확인", source_urls=[urls[-1]]),
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
        with self.assertRaisesRegex(RuntimeError, "TAVILY_API_KEY, LLM_PROVIDER, LLM_MODEL"):
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
