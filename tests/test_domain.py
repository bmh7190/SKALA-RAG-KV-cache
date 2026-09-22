"""도메인 평가의 현재 State 계약을 외부 API 없이 검증한다."""

import unittest
from unittest.mock import patch

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.domain.node import evaluate
from kv_cache_eval.features.domain.prompt import OUTPUT_SCHEMA, VERIFY_SCHEMA
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC


def evidence(id_, technology, claim):
    return {
        "id": id_, "technology": technology, "claim": claim, "excerpt": claim,
        "source": {"document": "paper.pdf", "url": None, "page": 3},
        "experiment": {"model": "Llama-2-7B", "workload": "long-context", "baseline": "FP16 KV cache"},
        "limitations": [], "verification_status": "source_checked",
    }


class DomainNodeTest(unittest.TestCase):
    def test_covers_all_criteria_and_keeps_cited_evidence(self):
        state = new_state()
        kivi_claim = "GPU memory decreased by 50%"
        inf_claim = "InfiniGen requires CPU memory offloading and KV prefetch."
        state["kivi_evidence"] = {"evidence": [evidence("kivi-1", "KIVI", kivi_claim)], "notes": []}
        state["infinigen_evidence"] = {"evidence": [evidence("inf-1", "InfiniGen", inf_claim)], "notes": []}

        def fake_invoke(messages, schema):
            if schema is OUTPUT_SCHEMA:
                return {"evaluations": [
                    {"technology": "KIVI", "criterion": "GPU 메모리 사용량",
                     "judgment": "GPU 메모리가 감소함", "score": None,
                     "rationale": "인용된 실험에서 50% 감소를 보고함",
                     "supports": [{"evidence_id": "kivi-1", "quote": kivi_claim}],
                     "measurement": {"kind": "reported_change", "magnitude": 50,
                                     "direction": "decrease",
                                     "source": {"evidence_id": "kivi-1", "quote": kivi_claim}},
                     "uncertainty": None},
                    {"technology": "InfiniGen", "criterion": "적용·운영 난이도",
                     "judgment": "CPU 메모리 오프로딩과 프리패치가 필요함", "score": None,
                     "rationale": "추가 운영 요소가 명시됨",
                     "supports": [{"evidence_id": "inf-1", "quote": inf_claim}],
                     "measurement": None, "uncertainty": None},
                ], "notes": []}
            self.assertIs(schema, VERIFY_SCHEMA)
            return {"reviews": [
                {"technology": "KIVI", "criterion": "GPU 메모리 사용량",
                 "supported": True, "measurement_supported": True,
                 "rubric_supported": False, "reason": "원문에 메모리 50% 감소가 명시됨"},
                {"technology": "InfiniGen", "criterion": "적용·운영 난이도",
                 "supported": True, "measurement_supported": False,
                 "rubric_supported": False, "reason": "원문에 추가 운영 요소가 명시됨"},
            ]}

        with patch("kv_cache_eval.features.domain.node.invoke_structured", side_effect=fake_invoke) as invoke:
            result = evaluate(state)

        evaluations = result["domain_eval"]["evaluations"]
        self.assertEqual(len(evaluations), 2 * len(DOMAIN_RUBRIC))
        self.assertEqual(invoke.call_count, 2)
        kivi = next(e for e in evaluations if (e["technology"], e["criterion"])
                    == ("KIVI", "GPU 메모리 사용량"))
        self.assertEqual((kivi["score"], kivi["evidence_ids"], kivi["basis_status"]),
                         (5, ["kivi-1"], "inferred"))
        infinigen = next(e for e in evaluations if (e["technology"], e["criterion"])
                        == ("InfiniGen", "적용·운영 난이도"))
        self.assertIsNone(infinigen["score"])
        self.assertEqual((infinigen["evidence_ids"], infinigen["basis_status"]),
                         (["inf-1"], "inferred"))
        self.assertEqual(sum(e["basis_status"] == "unverified" for e in evaluations), 10)
        self.assertEqual({e["id"] for e in result["domain_evidence"]["evidence"]}, {"kivi-1", "inf-1"})
        self.assertIn("KIVI:", result["domain_eval"]["text"])
        self.assertIn("InfiniGen:", result["domain_eval"]["text"])

    def test_missing_research_result_returns_unverified_rows_without_llm(self):
        with patch("kv_cache_eval.features.domain.node.invoke_structured") as invoke:
            result = evaluate(new_state())

        invoke.assert_not_called()
        evaluations = result["domain_eval"]["evaluations"]
        self.assertEqual(len(evaluations), 2 * len(DOMAIN_RUBRIC))
        self.assertTrue(all(e["score"] is None and e["basis_status"] == "unverified"
                            and not e["evidence_ids"] for e in evaluations))
        self.assertEqual(result["domain_evidence"]["evidence"], [])
        self.assertTrue(any("기술 조사 결과가 필요합니다" in note for note in result["domain_eval"]["notes"]))

    def test_hallucinated_evidence_id_is_dropped(self):
        state = new_state()
        state["kivi_evidence"] = {"evidence": [evidence("kivi-1", "KIVI", "설명")], "notes": []}
        state["infinigen_evidence"] = {"evidence": [], "notes": []}

        def fake_invoke(messages, schema):
            self.assertIs(schema, OUTPUT_SCHEMA)
            return {"evaluations": [
                {"technology": "KIVI", "criterion": "데이터 전송량",
                 "judgment": "전송량이 감소함", "score": 4,
                 "rationale": "존재하지 않는 근거를 인용함",
                 "supports": [{"evidence_id": "없는-id", "quote": "설명"}],
                 "measurement": None, "uncertainty": None},
            ], "notes": []}

        with patch("kv_cache_eval.features.domain.node.invoke_structured", side_effect=fake_invoke) as invoke:
            result = evaluate(state)

        self.assertEqual(invoke.call_count, 1)
        transfer = next(e for e in result["domain_eval"]["evaluations"]
                        if (e["technology"], e["criterion"]) == ("KIVI", "데이터 전송량"))
        self.assertIsNone(transfer["score"])
        self.assertIsNone(transfer["judgment"])
        self.assertEqual(transfer["evidence_ids"], [])
        self.assertEqual(transfer["basis_status"], "unverified")
        self.assertIn("근거 ID", transfer["uncertainty"])
        self.assertEqual(result["domain_evidence"]["evidence"], [])


if __name__ == "__main__":
    unittest.main()
