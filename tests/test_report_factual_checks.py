import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from test_report_quality import judge, ready_state

from kv_cache_eval.features.quality.node import evaluate
from kv_cache_eval.features.report.factual_checks import (
    FACTUAL_RULES,
    RECHECK_PAGES,
    collect_report_evidence,
)
from kv_cache_eval.features.report.node import BODY_FIELDS, export_pdf, write_report
from kv_cache_eval.features.technical_research.ingest import load_sources


class MandatoryFactualChecksTests(unittest.TestCase):
    def test_dataset_typo_cannot_pass_even_with_a_passing_judge(self):
        state = ready_state()
        state["report"]["sections"][3] = (
            "3. 평가 기준 및 방법",
            "WikiText2와 PTT로 측정했다. [kivi:a]",
        )

        def forbidden(_):
            raise AssertionError("typo must fail before the Judge")

        result = evaluate(state, chain=RunnableLambda(forbidden))["quality_result"]
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["groundedness"], "fail")
        self.assertIn("PTB", result["issues"][0]["required_action"])
        state.update(quality_result=result)
        with self.assertRaisesRegex(ValueError, "품질 통과"):
            export_pdf(state)

    def test_correct_dataset_and_latency_caveat_are_not_rejected(self):
        state = ready_state()
        state["report"]["sections"][3] = (
            "3. 평가 기준 및 방법",
            "PTB로 평가했다. 처리량 증가만으로 지연 감소를 입증할 수 없다. [kivi:a]",
        )
        self.assertTrue(evaluate(state, chain=judge())["quality_result"]["passed"])

    def test_writer_and_judge_receive_the_same_mandatory_rules(self):
        captured = {}

        def factory(schema, prompt, **kwargs):
            captured[kwargs["name"]] = prompt
            if kwargs["name"] == "report_quality":
                return judge()
            return RunnableLambda(
                lambda _: {
                    name: [{"text": "검증된 설명", "evidence_ids": ["kivi:a"]}]
                    for name in BODY_FIELDS
                }
            )

        state = ready_state()
        with (
            patch(
                "kv_cache_eval.features.report.node.structured_chain",
                side_effect=factory,
            ),
            patch(
                "kv_cache_eval.features.quality.node.structured_chain",
                side_effect=factory,
            ),
        ):
            state.update(write_report(state))
            evaluate(state)
        for prompt in captured.values():
            self.assertIn(FACTUAL_RULES, prompt)

    def test_unrepresented_documents_do_not_require_local_pdfs(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = collect_report_evidence(ready_state(), Path(directory))
        self.assertEqual(list(evidence), ["kivi:a"])

    def test_judge_assesses_current_report_without_stale_analysis(self):
        state = ready_state()
        state["evidence_gaps"] = [{"reason": "이전 검색에서는 미확인"}]
        state["domain_eval"] = {"evaluations": [{"judgment": "이전 판단"}]}

        def inspect_payload(inputs):
            payload = json.loads(inputs["payload"])
            self.assertEqual(set(payload), {"report", "evidence"})
            self.assertEqual(payload["report"], json.loads(json.dumps(state["report"])))
            return judge().invoke(inputs)

        self.assertTrue(
            evaluate(state, chain=RunnableLambda(inspect_payload))["quality_result"][
                "passed"
            ]
        )

    def test_represented_primary_missing_pdf_stops_factual_verification(self):
        state = ready_state()
        sources, _ = load_sources("KIVI")
        primary = next(
            source for source in sources if source.evidence_role == "primary"
        )
        state["kivi_evidence"]["evidence"][0]["source"]["document"] = (
            primary.citation_document
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                collect_report_evidence(state, Path(directory))

    def test_only_requested_pages_are_added_with_physical_page_citations(self):
        from types import SimpleNamespace

        state = ready_state()
        sources, _ = load_sources("KIVI")
        primary = next(
            source for source in sources if source.evidence_role == "primary"
        )
        state["kivi_evidence"]["evidence"][0]["source"]["document"] = (
            primary.citation_document
        )
        reader = SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda page=page: f"original page {page}")
                for page in range(1, primary.page_count + 1)
            ]
        )
        with (
            patch("kv_cache_eval.features.report.factual_checks.validate_sources"),
            patch("pypdf.PdfReader", return_value=reader),
        ):
            evidence = collect_report_evidence(state)
        added = [item for eid, item in evidence.items() if ":fact-check:" in eid]
        self.assertEqual(
            [item["source"]["page"] for item in added], list(RECHECK_PAGES["KIVI"])
        )
        for item in added:
            self.assertEqual(item["excerpt"], f"original page {item['source']['page']}")


if __name__ == "__main__":
    unittest.main()
