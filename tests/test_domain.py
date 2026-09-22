"""도메인 평가의 현재 공개 계약을 외부 LLM 없이 검증한다."""

import json
import unittest
from unittest.mock import patch

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.domain.node import evaluate
from kv_cache_eval.features.domain.prompt import OUTPUT_SCHEMA, VERIFY_SCHEMA
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC


CRITERION = "적용·운영 난이도"


def evidence(technology, claim):
    return {
        "id": technology.lower(), "technology": technology, "claim": claim, "excerpt": claim,
        "source": {"document": "paper.pdf", "url": None, "page": 3},
        "experiment": None, "limitations": [], "verification_status": "source_checked",
    }


class DomainNodeTest(unittest.TestCase):
    def test_covers_all_six_criteria_for_both_technologies(self):
        state = new_state()
        claims = {
            "KIVI": "KIVI 적용에는 라이브러리 설정 변경만 필요하다.",
            "InfiniGen": "InfiniGen 적용에는 별도 CPU 메모리 관리가 필요하다.",
        }
        for technology, key in (("KIVI", "kivi_evidence"), ("InfiniGen", "infinigen_evidence")):
            state[key] = {"evidence": [evidence(technology, claims[technology])], "notes": []}

        def fake_invoke(messages, schema):
            payload = json.loads(messages[1][1])
            self.assertEqual(payload["domain"], "GPU 기반 클라우드 LLM 서비스")
            if schema is OUTPUT_SCHEMA:
                return {"evaluations": [
                    {"technology": technology, "criterion": CRITERION,
                     "judgment": claims[technology], "score": score,
                     "rationale": "인용문에서 적용 조건을 확인했다.",
                     "supports": [{"evidence_id": technology.lower(), "quote": claims[technology]}],
                     "measurement": None, "uncertainty": None}
                    for technology, score in (("KIVI", 4), ("InfiniGen", 2))
                ], "notes": []}
            self.assertIs(schema, VERIFY_SCHEMA)
            return {"reviews": [
                {"technology": technology, "criterion": CRITERION, "supported": True,
                 "measurement_supported": False, "rubric_supported": True, "reason": "인용문과 기준에 부합"}
                for technology in ("KIVI", "InfiniGen")
            ]}

        with patch("kv_cache_eval.features.domain.node.invoke_structured", side_effect=fake_invoke) as llm:
            update = evaluate(state)
            result = update["domain_eval"]
        self.assertEqual(llm.call_count, 2)
        self.assertEqual({item["id"] for item in update["domain_evidence"]["evidence"]},
                         {"kivi", "infinigen"})
        self.assertIn("KIVI", result["text"])
        self.assertIn("InfiniGen", result["text"])
        self.assertEqual(len(result["evaluations"]), 2 * len(DOMAIN_RUBRIC))
        for technology, score in (("KIVI", 4), ("InfiniGen", 2)):
            row = next(item for item in result["evaluations"]
                       if item["technology"] == technology and item["criterion"] == CRITERION)
            self.assertEqual(row["score"], score)
            self.assertEqual(row["evidence_ids"], [technology.lower()])
            self.assertEqual(row["basis_status"], "inferred")
        unanswered = [row for row in result["evaluations"] if row["criterion"] != CRITERION]
        self.assertTrue(all(row["score"] is None and row["basis_status"] == "unverified" for row in unanswered))

    def test_missing_research_result_is_noted_without_llm(self):
        with patch("kv_cache_eval.features.domain.node.invoke_structured", side_effect=AssertionError("LLM called")):
            update = evaluate(new_state())
            result = update["domain_eval"]
        self.assertEqual(update["domain_evidence"]["evidence"], [])
        self.assertEqual(len(result["evaluations"]), 2 * len(DOMAIN_RUBRIC))
        self.assertTrue(all(row["basis_status"] == "unverified" for row in result["evaluations"]))
        self.assertTrue(any("조사 결과가 아직 제공되지" in note for note in result["notes"]))

    def test_hallucinated_evidence_id_is_dropped(self):
        state = new_state()
        claim = "KIVI 적용에는 라이브러리 설정 변경만 필요하다."
        state["kivi_evidence"] = {"evidence": [evidence("KIVI", claim)], "notes": []}
        state["infinigen_evidence"] = {"evidence": [], "notes": []}

        def fake_invoke(messages, schema):
            self.assertIs(schema, OUTPUT_SCHEMA)  # 잘못된 ID는 검토 호출 전에 제외한다.
            return {"evaluations": [{
                "technology": "KIVI", "criterion": CRITERION, "judgment": claim,
                "score": 4, "rationale": "목록에 없는 ID를 인용했다.",
                "supports": [{"evidence_id": "invented", "quote": claim}],
                "measurement": None, "uncertainty": None,
            }], "notes": []}

        with patch("kv_cache_eval.features.domain.node.invoke_structured", side_effect=fake_invoke) as llm:
            update = evaluate(state)
            result = update["domain_eval"]
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(update["domain_evidence"]["evidence"], [])
        row = next(item for item in result["evaluations"]
                   if item["technology"] == "KIVI" and item["criterion"] == CRITERION)
        self.assertIsNone(row["score"])
        self.assertEqual(row["evidence_ids"], [])
        self.assertEqual(row["basis_status"], "unverified")

    def test_rejected_rationale_is_excluded_from_text_and_domain_evidence(self):
        state = new_state()
        claim = "KIVI 적용에는 라이브러리 설정 변경만 필요하다."
        state["kivi_evidence"] = {"evidence": [evidence("KIVI", claim)], "notes": []}
        state["infinigen_evidence"] = {"evidence": [], "notes": []}

        def fake_invoke(messages, schema):
            if schema is OUTPUT_SCHEMA:
                return {"evaluations": [{
                    "technology": "KIVI", "criterion": CRITERION, "judgment": claim,
                    "score": 4, "rationale": "출처에 없는 운영 성과 주장",
                    "supports": [{"evidence_id": "kivi", "quote": claim}],
                    "measurement": None, "uncertainty": None,
                }], "notes": []}
            self.assertIs(schema, VERIFY_SCHEMA)
            return {"reviews": [{
                "technology": "KIVI", "criterion": CRITERION, "supported": False,
                "measurement_supported": False, "rubric_supported": False,
                "reason": "rationale가 근거에 없음",
            }]}

        with patch("kv_cache_eval.features.domain.node.invoke_structured", side_effect=fake_invoke):
            update = evaluate(state)
        row = next(item for item in update["domain_eval"]["evaluations"]
                   if item["technology"] == "KIVI" and item["criterion"] == CRITERION)
        self.assertEqual(row["basis_status"], "unverified")
        self.assertEqual(update["domain_evidence"]["evidence"], [])
        self.assertNotIn("출처에 없는 운영 성과 주장", update["domain_eval"]["text"])


if __name__ == "__main__":
    unittest.main()
