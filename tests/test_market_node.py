"""시장성 노드와 공통 근거 게이트 통합 테스트."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.market.analysis import CitedAssessment, MarketAnalysis
from kv_cache_eval.features.market.node import (
    MarketRuntimeConfig, build_market_node, evaluate, validate_runtime_config,
)
from kv_cache_eval.features.market.research import TavilyRateLimitError
from kv_cache_eval.graph.gates import check_evidence


class MarketNodeTest(unittest.TestCase):
    def _evaluate_after_429(self, state):
        def limited(query):
            raise TavilyRateLimitError("Tavily 429 요청 제한: 3회 시도 후 중단")

        with patch("kv_cache_eval.features.market.node.load_environment"), \
             patch("kv_cache_eval.features.market.node.validate_runtime_config",
                   return_value=MarketRuntimeConfig("openai", "fixture-model")), \
             patch("kv_cache_eval.features.market.node.tavily_search", return_value=limited), \
             patch("kv_cache_eval.features.market.node._openai_analyst", return_value=lambda *args: None):
            return evaluate(state)

    def test_429_reuses_previous_market_results_without_mutating_state(self):
        state = new_state()
        state["market_evidence"] = {"evidence": [{"id": "previous-evidence"}], "notes": ["이전 검색"]}
        state["market_eval"] = {"evaluations": [{"score": 2, "basis_status": "source_checked"}],
                                "notes": ["이전 평가"]}
        before = deepcopy(state)
        update = self._evaluate_after_429(state)
        self.assertEqual(state, before)
        self.assertEqual(update["market_evidence"]["evidence"], before["market_evidence"]["evidence"])
        self.assertEqual(update["market_eval"]["evaluations"], before["market_eval"]["evaluations"])
        self.assertIn("이번 실행의 이전 시장평가 재사용", update["market_eval"]["notes"][-1])
        self.assertIn("이번 재검색은 Tavily 429 실패", update["market_evidence"]["notes"][-1])

    def test_first_round_429_returns_six_unverified_results(self):
        state = new_state()
        update = self._evaluate_after_429(state)
        self.assertIsNone(state["market_evidence"])
        self.assertIsNone(state["market_eval"])
        self.assertEqual(update["market_evidence"]["evidence"], [])
        self.assertEqual(len(update["market_eval"]["evaluations"]), 6)
        self.assertTrue(all(row["judgment"] is None and row["score"] is None
                            and row["evidence_ids"] == [] and row["basis_status"] == "unverified"
                            for row in update["market_eval"]["evaluations"]))
        self.assertIn("429", update["market_eval"]["notes"][0])

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
