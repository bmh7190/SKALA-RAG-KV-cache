"""Response repair never turns bad citations into evidence or starts research."""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from test_domain import evidence, structured_fake

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.domain.node import evaluate
from kv_cache_eval.features.domain.prompt import OUTPUT_SCHEMA
from kv_cache_eval.features.supervisor.execution import worker

CLAIM = "At the same memory budget, KIVI gives 2.35× to 3.47× throughput."


def state(*criteria):
    s = new_state()
    s.update(
        next_agent="domain",
        step_count=1,
        retry_request={
            "technology": "KIVI",
            "criteria": list(criteria),
            "reason": "도메인 평가",
        },
    )
    s["kivi_evidence"] = {"evidence": [evidence("KIVI", CLAIM)], "notes": []}
    return s


def draft(criterion="처리량", quote=CLAIM):
    return {
        "technology": "KIVI",
        "criterion": criterion,
        "judgment": "논문은 조건부 처리량 범위를 보고한다",
        "score": None,
        "rationale": "같은 메모리 예산에서 2.35~3.47배로 보고됐으며 단일 값이 아니다",
        "supports": [{"evidence_id": "kivi", "quote": quote}],
        "measurement": None,
        "uncertainty": "운영 환경 일반화는 미확인",
    }


def reviews(payload, **flags):
    return {
        "reviews": [
            {
                "technology": d["technology"],
                "criterion": d["criterion"],
                "supported": True,
                "measurement_supported": False,
                "rubric_supported": False,
                "reason": "원문 범위는 판단을 지지하지만 점수는 미확인",
                **flags,
            }
            for d in payload["drafts"]
        ]
    }


class DomainRepairTests(unittest.TestCase):
    def test_changed_quote_repairs_only_failed_pair_with_identical_evidence(self):
        s = state("처리량", "모델 품질")
        calls = []

        def invoke(messages, schema):
            payload = json.loads(messages[1][1])
            if schema is not OUTPUT_SCHEMA:
                return reviews(payload)
            calls.append(payload)
            if len(calls) == 1:
                return {
                    "evaluations": [
                        draft(quote="KIVI gives ... throughput"),
                        draft("모델 품질"),
                    ],
                    "notes": [],
                }
            self.assertEqual(
                [(x["technology"], x["criterion"]) for x in payload["repair_targets"]],
                [("KIVI", "처리량")],
            )
            self.assertEqual(payload["evidence"], calls[0]["evidence"])
            self.assertEqual(list(payload["rubric"]), ["처리량"])
            return {"evaluations": [draft()], "notes": []}

        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            side_effect=structured_fake(invoke),
        ) as llm:
            result = evaluate(s)["domain_eval"]
        self.assertEqual(llm.call_count, 4)
        self.assertEqual(len(result["evaluations"]), 2)
        self.assertTrue(
            all(r["basis_status"] == "inferred" for r in result["evaluations"])
        )
        self.assertTrue(any("response_repair:" in n for n in result["notes"]))
        self.assertIsNone(s["domain_eval"])

    def test_range_as_percentage_requires_rewrite_before_qualitative_acceptance(self):
        s = state("처리량")
        count = 0

        def invoke(messages, schema):
            nonlocal count
            payload = json.loads(messages[1][1])
            if schema is not OUTPUT_SCHEMA:
                return reviews(payload)
            count += 1
            item = draft()
            if count == 1:
                item["measurement"] = {
                    "kind": "reported_change",
                    "magnitude": 135,
                    "direction": "increase",
                    "source": {"evidence_id": "kivi", "quote": CLAIM},
                }
                item["score"] = 5
            return {"evaluations": [item], "notes": []}

        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            side_effect=structured_fake(invoke),
        ):
            row = evaluate(s)["domain_eval"]["evaluations"][0]
        self.assertEqual(count, 2)
        self.assertEqual(row["basis_status"], "inferred")
        self.assertIsNone(row["score"])
        self.assertIn("2.35~3.47", row["rationale"])

    def test_real_evidence_gap_does_not_call_repair_or_review(self):
        item = {
            **draft(),
            "judgment": None,
            "score": None,
            "rationale": None,
            "supports": [],
            "uncertainty": "처리량 직접 측정 없음",
        }
        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            return_value={"evaluations": [item], "notes": []},
        ) as llm:
            result = worker("domain", evaluate)(state("처리량"), {})
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(result["last_result"]["status"], "needs_evidence")
        self.assertEqual(
            result["domain_eval"]["evaluations"][0]["failure_kind"], "evidence_gap"
        )

    def test_exhausted_bad_quote_is_execution_failure_not_research_request(self):
        response = {"evaluations": [draft(quote="invented quote")], "notes": []}
        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            side_effect=lambda *_: deepcopy(response),
        ) as llm:
            result = worker("domain", evaluate)(state("처리량"), {})
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result["last_result"]["status"], "failed")
        self.assertEqual(
            result["last_result"]["error"]["code"], "response_repair_exhausted"
        )
        self.assertEqual(result["last_result"]["gaps"], [])

    def test_timeout_is_not_retried_as_response_repair(self):
        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            side_effect=TimeoutError("temporary"),
        ) as llm:
            result = worker("domain", evaluate)(state("처리량"), {})
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(result["last_result"]["error"]["code"], "timeout")

    def test_supported_judgment_survives_unverified_qualitative_score(self):
        def invoke(messages, schema):
            if schema is OUTPUT_SCHEMA:
                return {
                    "evaluations": [{**draft("모델 품질"), "score": 4}],
                    "notes": [],
                }
            return reviews(json.loads(messages[1][1]))

        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            side_effect=structured_fake(invoke),
        ) as llm:
            row = evaluate(state("모델 품질"))["domain_eval"]["evaluations"][0]
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(row["basis_status"], "inferred")
        self.assertIsNone(row["score"])

    def test_unsupported_claim_can_be_rewritten_as_explicit_gap(self):
        drafts = 0

        def invoke(messages, schema):
            nonlocal drafts
            if schema is OUTPUT_SCHEMA:
                drafts += 1
                item = draft()
                if drafts == 2:
                    item.update(
                        judgment=None,
                        score=None,
                        rationale=None,
                        supports=[],
                        uncertainty="직접적인 판단 근거 부족",
                    )
                return {"evaluations": [item], "notes": []}
            return reviews(json.loads(messages[1][1]), supported=False)

        with patch(
            "kv_cache_eval.features.domain.node.invoke_structured",
            side_effect=structured_fake(invoke),
        ) as llm:
            result = worker("domain", evaluate)(state("처리량"), {})
        self.assertEqual(llm.call_count, 3)
        self.assertEqual(result["last_result"]["status"], "needs_evidence")
