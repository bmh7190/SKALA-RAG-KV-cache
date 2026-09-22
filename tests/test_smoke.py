"""API 없이 단일 조사 노드와 네 평가 노드의 Graph 배선을 확인한다."""

import unittest
from unittest.mock import patch

from kv_cache_eval.common.state import new_state
from kv_cache_eval.graph import build_graph
from kv_cache_eval.graph.gates import check_evidence, route_after_check


class SmokeTest(unittest.TestCase):
    def test_fresh_state_and_empty_evidence_are_distinct(self):
        first = new_state()
        second = new_state()
        self.assertIsNone(first["evidence_gaps"])
        self.assertIsNone(first["kivi_evidence"])
        self.assertIsNone(first["maturity_eval"])
        first["domain_and_criteria"]["domain"] = "changed"
        self.assertEqual(second["domain_and_criteria"]["domain"], "GPU 기반 클라우드 LLM 서비스")
        first.update({"kivi_evidence": {"evidence": [], "notes": []}, "infinigen_evidence": {"evidence": [], "notes": []}})
        gaps = check_evidence(first)["evidence_gaps"]
        self.assertTrue(gaps)
        self.assertEqual(route_after_check({**first, "evidence_gaps": gaps}), "retry")

    def test_evaluations_start_after_both_research_results(self):
        events: list[str] = []

        def research(state):
            events.append("technical_research")
            empty = {"evidence": [], "notes": []}
            return {"kivi_evidence": empty, "infinigen_evidence": empty}

        def evaluate(key):
            def node(state):
                self.assertIsNotNone(state["kivi_evidence"])
                self.assertIsNotNone(state["infinigen_evidence"])
                events.append(key)
                return {key: {"evaluations": [], "notes": []}}
            return node

        def checked(state):
            self.assertEqual(len(events), 5)
            self.assertEqual(events[0], "technical_research")
            for key in ("maturity_eval", "market_eval", "stakeholder_eval", "domain_eval"):
                self.assertIsNotNone(state[key])
            return {"evidence_gaps": []}  # 배선 테스트 전용. 실제 게이트는 빈 근거를 거부한다.

        graph = build_graph({
            "technical_research": research,
            "maturity": evaluate("maturity_eval"),
            "market": evaluate("market_eval"),
            "stakeholders": evaluate("stakeholder_eval"),
            "domain": evaluate("domain_eval"),
            "evidence_check": checked,
            "synthesis": lambda state: {"synthesis": {"perspective_differences": [], "tradeoffs": [], "application_conditions": [], "unresolved_gaps": [], "cited_evidence_ids": []}},
            "report": lambda state: {"report": {"sections": [("SUMMARY", "테스트 전용"), ("REFERENCE", "테스트 전용")], "cited_evidence_ids": []}},
        })
        final = graph.invoke(new_state(max_research_rounds=0))
        self.assertEqual(final["evidence_gaps"], [])
        self.assertEqual(final["report"]["sections"][0][0], "SUMMARY")

    def test_retry_uses_same_research_node_and_keeps_unresolved_gaps(self):
        rounds = []

        def research(state, *, user_question):
            rounds.append((state["research_round"], user_question))
            if state["research_round"]:
                self.assertIsNotNone(state["kivi_evidence"])
                self.assertIsNotNone(state["infinigen_evidence"])
            empty = {"evidence": [], "notes": []}
            return {"kivi_evidence": empty, "infinigen_evidence": empty}

        def evaluate(key):
            return lambda state: {key: {"evaluations": [], "notes": []}}

        def synthesize(state):
            self.assertTrue(state["evidence_gaps"])
            return {"synthesis": {"perspective_differences": [], "tradeoffs": [], "application_conditions": [], "unresolved_gaps": state["evidence_gaps"], "cited_evidence_ids": []}}

        with patch("kv_cache_eval.graph.workflow.research", side_effect=research):
            graph = build_graph({
                "maturity": evaluate("maturity_eval"),
                "market": evaluate("market_eval"),
                "stakeholders": evaluate("stakeholder_eval"),
                "domain": evaluate("domain_eval"),
                "synthesis": synthesize,
                "report": lambda state: {"report": {"sections": [], "cited_evidence_ids": []}},
            }, user_question="같은 질문")
            final = graph.invoke(new_state(max_research_rounds=1))
        self.assertEqual(rounds, [(0, "같은 질문"), (1, "같은 질문")])
        self.assertEqual(final["research_round"], 1)
        self.assertTrue(final["evidence_gaps"])
        self.assertEqual(final["synthesis"]["unresolved_gaps"], final["evidence_gaps"])

    def test_default_graph_compiles_without_optional_llm_import(self):
        # 기본 노드는 컴파일만 확인한다. 실행에는 외부 검색·LLM 호출이 포함된다.
        self.assertIsNotNone(build_graph())


if __name__ == "__main__":
    unittest.main()
