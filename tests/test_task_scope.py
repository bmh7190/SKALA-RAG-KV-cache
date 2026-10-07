import unittest
from unittest.mock import patch

from kv_cache_eval.common.evidence import collect_evidence
from kv_cache_eval.common.state import new_state
from kv_cache_eval.common.tasks import merge_evaluation
from kv_cache_eval.features.domain.node import evaluate
from kv_cache_eval.features.technical_research.prompts import questions_for_state


class ScopeTests(unittest.TestCase):
    def test_research_retry_only_asks_requested_question(self):
        s = new_state()
        s["retry_request"] = {
            "technology": "InfiniGen",
            "criteria": ["experiment_conditions"],
            "reason": "조건 누락",
        }
        questions = questions_for_state(s, "InfiniGen")
        self.assertEqual([q.id for q in questions], ["experiment_conditions"])
        self.assertIn("조건 누락", questions[0].text)

    def test_partial_evaluation_preserves_other_technology(self):
        kivi = {"technology": "KIVI", "criterion": "처리량", "score": 4}
        old = {"technology": "InfiniGen", "criterion": "처리량", "score": None}
        new = {**old, "score": 3}
        s = {"domain_eval": {"evaluations": [kivi, old], "notes": []}}
        result = merge_evaluation(s, "domain_eval", {"evaluations": [new], "notes": []})
        self.assertEqual(result["evaluations"], [kivi, new])
        self.assertIsNone(s["domain_eval"]["evaluations"][1]["score"])

    def test_domain_partial_request_does_not_call_other_technology(self):
        s = new_state()
        s["retry_request"] = {
            "technology": "InfiniGen",
            "criteria": ["처리량"],
            "reason": "조건 누락",
        }
        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            side_effect=AssertionError("no evidence"),
        ):
            result = evaluate(s)
        self.assertEqual(
            [
                (x["technology"], x["criterion"])
                for x in result["domain_eval"]["evaluations"]
            ],
            [("InfiniGen", "처리량")],
        )

    def test_verified_evidence_optional_nulls_are_equivalent(self):
        item = {
            "id": "a",
            "claim": "A",
            "source": {"document": "a.pdf"},
            "verification_status": "source_checked",
            "experiment": {},
        }
        s = {
            "kivi_evidence": {"evidence": [item]},
            "domain_evidence": {"evidence": [{**item, "experiment": {"model": None}}]},
        }
        self.assertEqual(list(collect_evidence(s)), ["a"])
        s["domain_evidence"]["evidence"][0]["claim"] = "B"
        with self.assertRaisesRegex(ValueError, "상충"):
            collect_evidence(s)
