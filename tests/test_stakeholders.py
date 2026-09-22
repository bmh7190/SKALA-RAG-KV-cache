"""이해관계자 평가 노드 단위 테스트.

실제 LLM을 호출하지 않는다: evaluate()에 llm_call을 주입해 langchain-openai 설치나
API 키 없이도(오프라인 smoke test와 동일한 기조) 노드 로직만 검증한다.
"""

import unittest

from pydantic import ValidationError

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
        self.assertEqual(kivi_operator["criterion"], first.criterion)  # criterion은 그룹으로부터 자동 채워짐
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

    def test_no_verified_evidence_skips_llm_call(self):
        state = new_state()
        # 근거는 있지만 아직 검증(source_checked)되지 않은 경우 — LLM을 호출하지 않고 바로 보류해야 함.
        state["kivi_evidence"] = {
            "evidence": [_make_evidence("kivi-1", "KIVI", "미검증 주장", verification_status="unverified")],
            "notes": [],
        }
        state["infinigen_evidence"] = {"evidence": [], "notes": []}

        def fake_llm(*, technology, evidence):
            raise AssertionError("검증된 근거가 없으면 LLM을 호출하지 않아야 한다")

        result = evaluate(state, llm_call=fake_llm)["stakeholder_eval"]
        kivi_entries = [e for e in result["evaluations"] if e["technology"] == "KIVI"]
        self.assertEqual(len(kivi_entries), len(STAKEHOLDER_CRITERIA))
        self.assertTrue(all(e["score"] is None for e in kivi_entries))

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

    def test_unknown_group_name_is_rejected_at_construction(self):
        # 모델이 5개 그룹 이름 중 하나가 아닌 값을 반환하면 pydantic 단계에서 즉시 걸려야 한다
        # (자유 텍스트였을 때는 이게 통과돼서 항목이 중복 생성되는 문제가 있었음).
        with self.assertRaises(ValidationError):
            _StakeholderAssessment(
                stakeholder_group=STAKEHOLDER_CRITERIA[0].group + " ",  # 사소한 변형
                judgment="x",
                score=4,
                rationale=None,
                evidence_ids=[],
                uncertainty=None,
                basis_status="source_checked",
            )

    def test_duplicate_valid_group_is_ignored_after_the_first(self):
        state = new_state()
        state["kivi_evidence"] = {"evidence": [_make_evidence("kivi-1", "KIVI", "설명")], "notes": []}
        state["infinigen_evidence"] = {"evidence": [], "notes": []}

        first = STAKEHOLDER_CRITERIA[0]

        def dup_llm(*, technology, evidence):
            if technology != "KIVI":
                return _StakeholderAssessmentBatch(assessments=[], notes=[])
            return _StakeholderAssessmentBatch(
                assessments=[
                    _StakeholderAssessment(
                        stakeholder_group=first.group, judgment="a", score=3,
                        rationale=None, evidence_ids=["kivi-1"], uncertainty=None, basis_status="source_checked",
                    ),
                    _StakeholderAssessment(
                        stakeholder_group=first.group, judgment="b", score=5,
                        rationale=None, evidence_ids=["kivi-1"], uncertainty=None, basis_status="source_checked",
                    ),
                ],
                notes=[],
            )

        result = evaluate(state, llm_call=dup_llm)["stakeholder_eval"]
        kivi_entries = [e for e in result["evaluations"] if e["technology"] == "KIVI"]
        self.assertEqual(len(kivi_entries), len(STAKEHOLDER_CRITERIA))

        matches = [e for e in kivi_entries if e["stakeholder_group"] == first.group]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["score"], 3)  # 첫 번째 것만 채택
        self.assertTrue(any("중복 평가를 무시함" in n for n in result["notes"]))

    def test_llm_failure_for_one_technology_does_not_crash_the_other(self):
        state = new_state()
        state["kivi_evidence"] = {"evidence": [_make_evidence("kivi-1", "KIVI", "설명")], "notes": []}
        state["infinigen_evidence"] = {"evidence": [_make_evidence("inf-1", "InfiniGen", "설명")], "notes": []}

        def flaky_llm(*, technology, evidence):
            if technology == "KIVI":
                raise RuntimeError("일시적인 API 오류 가정")
            return _StakeholderAssessmentBatch(
                assessments=[
                    _StakeholderAssessment(
                        stakeholder_group=STAKEHOLDER_CRITERIA[0].group,
                        judgment="ok",
                        score=5,
                        rationale=None,
                        evidence_ids=["inf-1"],
                        uncertainty=None,
                        basis_status="source_checked",
                    )
                ],
                notes=[],
            )

        result = evaluate(state, llm_call=flaky_llm)["stakeholder_eval"]

        kivi_entries = [e for e in result["evaluations"] if e["technology"] == "KIVI"]
        self.assertEqual(len(kivi_entries), len(STAKEHOLDER_CRITERIA))
        self.assertTrue(all(e["score"] is None for e in kivi_entries))
        self.assertTrue(any("LLM 호출 실패" in n for n in result["notes"]))

        infinigen_first = next(
            e
            for e in result["evaluations"]
            if e["technology"] == "InfiniGen" and e["stakeholder_group"] == STAKEHOLDER_CRITERIA[0].group
        )
        self.assertEqual(infinigen_first["score"], 5)


if __name__ == "__main__":
    unittest.main()
