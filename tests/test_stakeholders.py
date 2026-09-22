"""이해관계자 평가 노드 단위 테스트.

실제 LLM을 호출하지 않는다: evaluate()에 llm_call을 주입해 langchain-openai 설치나
API 키 없이도(오프라인 smoke test와 동일한 기조) 노드 로직만 검증한다.
"""

import unittest

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.stakeholders.criteria import STAKEHOLDER_CRITERIA
from kv_cache_eval.features.stakeholders.node import (
    _StakeholderAssessment,
    _StakeholderAssessmentBatch,
    evaluate,
)


def _make_evidence(id_, technology, claim, verification_status="source_checked"):
    return {
        "id": id_,
        "technology": technology,
        "claim": claim,
        "excerpt": claim,
        "source": {"document": "paper.pdf", "url": None, "page": 3},
        "experiment": {"model": "Llama-2-7B", "workload": "long-context", "baseline": "FP16 KV cache"},
        "limitations": [],
        "verification_status": verification_status,
    }


class StakeholderNodeTest(unittest.TestCase):
    def test_covers_all_five_groups_for_both_technologies(self):
        state = new_state()
        state["kivi_evidence"] = {
            "evidence": [_make_evidence("kivi-1", "KIVI", "2bit 양자화로 peak memory 61% 감소")],
            "notes": [],
        }
        state["infinigen_evidence"] = {
            "evidence": [_make_evidence("inf-1", "InfiniGen", "필수 KV만 프리패치, CPU 메모리 오프로딩")],
            "notes": [],
        }

        first = STAKEHOLDER_CRITERIA[0]

        def fake_llm(*, technology, evidence):
            assessments = [
                _StakeholderAssessment(
                    stakeholder_group=first.group,
                    criterion=first.criterion,
                    judgment="운영 효율 이점이 확인됨",
                    score=4,
                    rationale="근거에서 메모리 절감을 직접 확인",
                    evidence_ids=[e["id"] for e in evidence],
                    uncertainty=None,
                    basis_status="source_checked",
                )
            ]
            return _StakeholderAssessmentBatch(assessments=assessments, notes=[])

        result = evaluate(state, llm_call=fake_llm)["stakeholder_eval"]

        self.assertEqual(len(result["evaluations"]), 2 * len(STAKEHOLDER_CRITERIA))

        kivi_operator = next(
            e for e in result["evaluations"] if e["technology"] == "KIVI" and e["stakeholder_group"] == first.group
        )
        self.assertEqual(kivi_operator["score"], 4)
        self.assertEqual(kivi_operator["criterion"], first.criterion)
        self.assertEqual(kivi_operator["evidence_ids"], ["kivi-1"])
        self.assertEqual(kivi_operator["basis_status"], "source_checked")

        # 나머지 4개 그룹은 모델이 반환하지 않았으므로 판단 보류(미확인) 상태여야 한다.
        unanswered = [e for e in result["evaluations"] if e["stakeholder_group"] != first.group]
        self.assertEqual(len(unanswered), 2 * len(STAKEHOLDER_CRITERIA) - 2)
        self.assertTrue(all(e["score"] is None and e["judgment"] is None for e in unanswered))
        self.assertTrue(all(e["evidence_ids"] == [] for e in unanswered))
        self.assertTrue(all(e["basis_status"] == "unverified" for e in unanswered))

    def test_missing_research_result_is_noted_not_crashed(self):
        state = new_state()  # kivi_evidence / infinigen_evidence 모두 아직 None

        def fake_llm(*, technology, evidence):
            raise AssertionError("근거가 없으면 LLM을 호출하지 않아야 한다")

        result = evaluate(state, llm_call=fake_llm)["stakeholder_eval"]
        self.assertEqual(result["evaluations"], [])
        self.assertTrue(any("기술조사 결과가 아직 없어" in n for n in result["notes"]))

    def test_hallucinated_evidence_id_is_dropped_and_downgraded(self):
        state = new_state()
        state["kivi_evidence"] = {"evidence": [_make_evidence("kivi-1", "KIVI", "설명")], "notes": []}
        state["infinigen_evidence"] = {"evidence": [], "notes": []}

        competitor = STAKEHOLDER_CRITERIA[3]  # 경쟁 기술 진영

        def fake_llm(*, technology, evidence):
            if technology != "KIVI":
                return _StakeholderAssessmentBatch(assessments=[], notes=[])
            return _StakeholderAssessmentBatch(
                assessments=[
                    _StakeholderAssessment(
                        stakeholder_group=competitor.group,
                        criterion=competitor.criterion,
                        judgment="차별성이 있음",
                        score=4,
                        rationale="목록에 없는 id를 인용함",
                        evidence_ids=["존재하지-않는-id"],
                        uncertainty=None,
                        basis_status="inferred",
                    )
                ],
                notes=[],
            )

        result = evaluate(state, llm_call=fake_llm)["stakeholder_eval"]
        entry = next(
            e for e in result["evaluations"] if e["technology"] == "KIVI" and e["stakeholder_group"] == competitor.group
        )
        self.assertIsNone(entry["score"])
        self.assertIsNone(entry["judgment"])
        self.assertEqual(entry["evidence_ids"], [])
        self.assertEqual(entry["basis_status"], "unverified")
        self.assertTrue(any("확인되지 않은 근거 ID" in n for n in result["notes"]))


if __name__ == "__main__":
    unittest.main()
