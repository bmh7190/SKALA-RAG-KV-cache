"""네트워크/API 키 없이 State 초기값과 실제 그래프 합류만 확인한다."""

import unittest

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

    def test_parallel_joins_and_partial_updates(self):
        events: list[str] = []

        def research(key):
            def node(state):
                events.append(key)
                return {key: {"evidence": [], "notes": []}}
            return node

        def evaluate(key):
            def node(state):
                self.assertIsNotNone(state["kivi_evidence"])
                self.assertIsNotNone(state["infinigen_evidence"])
                events.append(key)
                return {key: {"evaluations": [], "notes": []}}
            return node

        def checked(state):
            self.assertEqual(len(events), 6)
            for key in ("maturity_eval", "market_eval", "stakeholder_eval", "domain_eval"):
                self.assertIsNotNone(state[key])
            return {"evidence_gaps": []}  # 배선 테스트 전용. 실제 게이트는 빈 근거를 거부한다.

        graph = build_graph({
            "research_kivi": research("kivi_evidence"),
            "research_infinigen": research("infinigen_evidence"),
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

    def test_real_gate_keeps_unresolved_gaps_on_exhaustion(self):
        def research(key):
            return lambda state: {key: {"evidence": [], "notes": []}}

        def evaluate(key):
            return lambda state: {key: {"evaluations": [], "notes": []}}

        def synthesize(state):
            self.assertTrue(state["evidence_gaps"])
            return {"synthesis": {"perspective_differences": [], "tradeoffs": [], "application_conditions": [], "unresolved_gaps": state["evidence_gaps"], "cited_evidence_ids": []}}

        graph = build_graph({
            "research_kivi": research("kivi_evidence"),
            "research_infinigen": research("infinigen_evidence"),
            "maturity": evaluate("maturity_eval"),
            "market": evaluate("market_eval"),
            "stakeholders": evaluate("stakeholder_eval"),
            "domain": evaluate("domain_eval"),
            "synthesis": synthesize,
            "report": lambda state: {"report": {"sections": [], "cited_evidence_ids": []}},
        })
        final = graph.invoke(new_state(max_research_rounds=1))
        self.assertEqual(final["research_round"], 1)
        self.assertTrue(final["evidence_gaps"])
        self.assertEqual(final["synthesis"]["unresolved_gaps"], final["evidence_gaps"])

    def test_default_function_node_reports_unimplemented(self):
        with self.assertRaises(NotImplementedError):
            build_graph().invoke(new_state())


if __name__ == "__main__":
    unittest.main()
