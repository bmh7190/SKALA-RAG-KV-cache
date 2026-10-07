"""실제 실패 원인: 허구 지적, 수정 범위 소실, 미반영 재작성 회귀 검증."""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from test_report_quality import judge, ready_state

from kv_cache_eval.common.errors import classify_error
from kv_cache_eval.features.quality.node import evaluate
from kv_cache_eval.features.report.feedback import ReportValidationError, issue_sections
from kv_cache_eval.features.report.node import write_report
from kv_cache_eval.features.supervisor.node import _dispatch_report_stage
from kv_cache_eval.features.supervisor.retries import failed_work

QUOTE = "검증용 예시이며 실제 평가가 아닙니다."


def verdict(**overrides):
    result = judge().invoke({})
    finding = dict(
        section="2. 기술 선정 및 개요",
        kind="claim",
        quote=QUOTE,
        evidence_ids=["kivi:a"],
        reason="발췌 범위를 넘는 단정",
        required_action="발췌 범위를 명시해 수정한다",
    )
    finding.update(overrides)
    result["groundedness"].update(status="fail", findings=[finding])
    return result


def revision_state():
    state = ready_state()
    state.update(evaluate(state, chain=RunnableLambda(lambda _: verdict())))
    state.update(report_revision=1, next_agent="report")
    return state


def output(state, replacements):
    return {
        "sections": [
            {"title": title, "content": replacements.get(title, text)}
            for title, text in state["report"]["sections"]
        ],
        "cited_evidence_ids": ["kivi:a"],
    }


class QualityFeedbackTests(unittest.TestCase):
    def test_runtime_judge_schema_restricts_evidence_ids(self):
        from pydantic import ValidationError

        from kv_cache_eval.features.quality.judgment import judgment_schema

        schema = judgment_schema({"kivi:a": {}})
        schema.model_validate(verdict())
        with self.assertRaises(ValidationError):
            schema.model_validate(verdict(evidence_ids=["kivi:invented"]))

    def test_native_langchain_model_response_is_accepted(self):
        from kv_cache_eval.features.quality.judgment import judgment_schema

        response = judgment_schema({"kivi:a": {}}).model_validate(judge().invoke({}))
        self.assertTrue(
            evaluate(ready_state(), chain=RunnableLambda(lambda _: response))[
                "quality_result"
            ]["passed"]
        )

    def test_multiple_findings_keep_individual_top_level_sections(self):
        raw = verdict()
        raw["groundedness"]["findings"].append(
            {**raw["groundedness"]["findings"][0], "section": "SUMMARY"}
        )
        state = ready_state()
        state.update(evaluate(state, chain=RunnableLambda(lambda _: raw)))
        state.update(completed_agents=["report", "quality"], agent_calls={"report": 1})
        update = _dispatch_report_stage(state, set(state["completed_agents"]))
        self.assertEqual(
            update["retry_request"]["criteria"], ["SUMMARY", "2. 기술 선정 및 개요"]
        )
        self.assertEqual(update["report_revision"], 1)
        self.assertEqual(update["quality_result"]["issues"][0]["quote"], QUOTE)

    def test_legacy_composite_sections_are_not_lost(self):
        self.assertEqual(
            issue_sections(
                [
                    {
                        "section": "SUMMARY / 2. 기술 선정 및 개요 / 4.1 기술 성숙도 / 6. 한계점"
                    }
                ]
            ),
            ["SUMMARY", "2. 기술 선정 및 개요", "4. 관점별 평가 결과", "6. 한계점"],
        )

    def test_made_up_or_wrong_section_quotes_reject_judge_not_report(self):
        for quote in ("본문에 없는 1.63~32.93배 성능", ""):
            with self.subTest(quote=quote):
                state = ready_state()
                before = deepcopy(state)
                with self.assertRaises(ValueError) as raised:
                    evaluate(
                        state, chain=RunnableLambda(lambda _: verdict(quote=quote))
                    )
                state.update(agent_calls={"quality": 1, "report": 1})
                update = failed_work(
                    state,
                    {"agent": "quality", "error": classify_error(raised.exception)},
                )
                self.assertEqual(update["next_agent"], "quality")
                self.assertEqual(update["report_revision"], 0)
                self.assertIn("실제 본문", update["retry_request"]["reason"])
                self.assertEqual(state["report"], before["report"])

    def test_invalid_evidence_or_composite_model_section_is_rejected(self):
        for fields in (
            {"evidence_ids": ["missing:id"]},
            {"section": "SUMMARY / 6. 한계점"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                evaluate(
                    ready_state(), chain=RunnableLambda(lambda _: verdict(**fields))
                )

    def test_pass_with_findings_and_fail_without_findings_are_rejected(self):
        for raw in (verdict(), judge().invoke({})):
            raw["groundedness"]["status"] = (
                "pass" if raw["groundedness"]["findings"] else "fail"
            )
            with self.assertRaises(ValueError):
                evaluate(ready_state(), chain=RunnableLambda(lambda _: raw))

    def test_no_quote_allowed_only_for_missing_coverage(self):
        with self.assertRaises(ValueError):
            evaluate(
                ready_state(),
                chain=RunnableLambda(
                    lambda _: verdict(kind="missing_content", quote="")
                ),
            )
        self.assertFalse(
            evaluate(ready_state(), chain=judge("fail"))["quality_result"]["passed"]
        )

    def test_judge_receives_task_scope_and_its_own_validation_feedback(self):
        state = ready_state()
        state.update(
            next_agent="quality", retry_request={"reason": "본문을 정확히 인용하세요"}
        )

        def inspect(inputs):
            payload = json.loads(inputs["payload"])
            self.assertEqual(payload["question"], state["question"])
            self.assertEqual(
                payload["domain"], json.loads(json.dumps(state["domain_and_criteria"]))
            )
            self.assertEqual(
                payload["validation_feedback"], state["retry_request"]["reason"]
            )
            return judge().invoke(inputs)

        self.assertTrue(
            evaluate(state, chain=RunnableLambda(inspect))["quality_result"]["passed"]
        )

    def test_rewrite_with_unchanged_problem_sentence_is_rejected(self):
        state = revision_state()
        raw = output(
            state, {"2. 기술 선정 및 개요": QUOTE + "   [kivi:a]\n표현만 추가"}
        )
        with self.assertRaisesRegex(ReportValidationError, "그대로"):
            write_report(state, chain=RunnableLambda(lambda _: raw))

    def test_rewrite_preserves_unflagged_sections_and_passes_targets_to_writer(self):
        state = revision_state()
        before = deepcopy(state)
        raw = output(
            state,
            {
                "2. 기술 선정 및 개요": "수집한 발췌의 범위만 확인했다. [kivi:a]",
                "SUMMARY": "원치 않는 변경 [kivi:a]",
            },
        )

        def inspect(inputs):
            payload = json.loads(inputs["payload"])
            self.assertEqual(payload["revision_sections"], ["2. 기술 선정 및 개요"])
            self.assertEqual(payload["quality_feedback"]["issues"][0]["quote"], QUOTE)
            return raw

        result = write_report(state, chain=RunnableLambda(inspect))["report"]
        self.assertEqual(
            dict(result["sections"])["SUMMARY"],
            dict(before["report"]["sections"])["SUMMARY"],
        )
        self.assertIn("수집한 발췌", dict(result["sections"])["2. 기술 선정 및 개요"])
        self.assertEqual(state, before)

    def test_runtime_rewrite_schema_only_requests_flagged_sections(self):
        state = revision_state()

        def factory(schema, prompt, **kwargs):
            self.assertEqual(set(schema.model_fields), {"cover", "technology_overview"})
            return RunnableLambda(
                lambda _: {
                    "cover": {"title": "수정 제목", "subtitle": "", "scope": ""},
                    "technology_overview": [
                        {
                            "text": "실제 원문의 조건을 확인했다.",
                            "evidence_ids": ["kivi:a"],
                        }
                    ],
                }
            )

        with patch(
            "kv_cache_eval.features.report.node.structured_chain", side_effect=factory
        ):
            result = write_report(state)["report"]
        old = dict(state["report"]["sections"])
        for title, text in result["sections"]:
            if title != "2. 기술 선정 및 개요":
                self.assertEqual(text, old[title])
        self.assertEqual(result["cited_evidence_ids"], ["kivi:a"])

    def test_unchanged_missing_content_is_rejected(self):
        state = ready_state()
        state.update(evaluate(state, chain=judge("fail")))
        state["report_revision"] = 1
        with self.assertRaisesRegex(ReportValidationError, "반영되지"):
            write_report(state, chain=RunnableLambda(lambda _: output(state, {})))
