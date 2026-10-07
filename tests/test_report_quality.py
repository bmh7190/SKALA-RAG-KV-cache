import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from kv_cache_eval.common.evidence import collect_evidence
from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.quality.node import CRITERIA, evaluate
from kv_cache_eval.features.report.node import (
    REPORT_SECTION_TITLES,
    ReportTooLong,
    export_pdf,
    printable_report,
    write_report,
)


def ready_state():
    s = new_state()
    s["kivi_evidence"] = {
        "evidence": [
            {
                "id": "kivi:a",
                "technology": "KIVI",
                "claim": "메모리 절감",
                "excerpt": "메모리 절감",
                "source": {"document": "paper", "url": None, "page": 1},
                "verification_status": "source_checked",
                "experiment": None,
                "limitations": [],
            }
        ],
        "notes": [],
    }
    s["report"] = {
        "sections": [
            (
                t,
                ""
                if t == "REFERENCE"
                else "검증용 예시이며 실제 평가가 아닙니다. [kivi:a]",
            )
            for t in REPORT_SECTION_TITLES
        ],
        "cited_evidence_ids": ["kivi:a"],
    }
    s["synthesis"] = {}
    for key in ("maturity_eval", "market_eval", "stakeholder_eval", "domain_eval"):
        s[key] = {"evaluations": [], "notes": []}
    return s


def judge(coverage="pass"):
    return RunnableLambda(
        lambda _: {
            name: {
                "status": coverage if name == "perspective_coverage" else "pass",
                "reason": "오프라인 검증용",
                "section": "4. 관점별 평가 결과",
                "required_action": "누락 관점 보완",
            }
            for name in CRITERIA
        }
    )


class PrintableCitationTests(unittest.TestCase):
    def setUp(self):
        self.report = ready_state()["report"]
        original = collect_evidence(ready_state())["kivi:a"]
        self.evidence = {
            "kivi:a": original,
            "kivi:b": {**deepcopy(original), "id": "kivi:b", "claim": "다른 근거"},
            "kivi:c": {
                **deepcopy(original),
                "id": "kivi:c",
                "source": {**original["source"], "page": 2},
            },
        }

    def render(self, text):
        self.report["sections"] = [("SUMMARY", text), ("REFERENCE", "")]
        return printable_report(self.report, self.evidence)

    def test_same_source_page_collapses_after_numbering(self):
        for separator in (" ", "", "\t"):
            with self.subTest(separator=separator):
                text = f"설명 [kivi:a]{separator}[kivi:b] [kivi:c] [kivi:a]."
                result = self.render(text)
                self.assertEqual(result["sections"][0][1], "설명 [1] [2].")
                self.assertEqual(
                    result["sections"][1][1], "[1] paper, p. 1\n[2] paper, p. 2"
                )

    def test_repeated_id_in_one_group_is_printed_once(self):
        self.assertEqual(
            self.render("설명 [kivi:a] [kivi:a].")["sections"][0][1], "설명 [1]."
        )

    def test_distinct_pages_and_separate_claims_keep_their_citations(self):
        text = "첫 주장 [kivi:a] [kivi:c]. 두 번째 주장 [kivi:b]."
        self.assertEqual(
            self.render(text)["sections"][0][1], "첫 주장 [1] [2]. 두 번째 주장 [1]."
        )

    def test_line_and_paragraph_boundaries_are_preserved(self):
        for separator in ("\n", "\n\n", "\r\n"):
            with self.subTest(separator=separator):
                text = f"[kivi:a]{separator}[kivi:b]"
                self.assertEqual(
                    self.render(text)["sections"][0][1], f"[1]{separator}[1]"
                )

    def test_internal_report_and_evidence_are_not_modified(self):
        self.report["sections"] = [("SUMMARY", "설명 [kivi:a] [kivi:b].")]
        self.report["cited_evidence_ids"] = ["kivi:a", "kivi:b"]
        before_report, before_evidence = deepcopy(self.report), deepcopy(self.evidence)
        printable_report(self.report, self.evidence)
        self.assertEqual(self.report, before_report)
        self.assertEqual(self.evidence, before_evidence)

    def test_unknown_id_is_rejected_before_formatting(self):
        with self.assertRaisesRegex(ValueError, "확인되지 않은 근거 ID"):
            self.render("설명 [kivi:a] [unknown].")


class ReportQualityTests(unittest.TestCase):
    def test_pass_and_revision_bound_to_current_draft(self):
        s = ready_state()
        s["report_revision"] = 2
        result = evaluate(s, chain=judge())["quality_result"]
        self.assertTrue(result["passed"])
        self.assertEqual(result["report_revision"], 2)

    def test_content_failure_cannot_be_passed(self):
        result = evaluate(ready_state(), chain=judge("fail"))["quality_result"]
        self.assertFalse(result["passed"])
        self.assertEqual(result["issues"][0]["criterion"], "perspective_coverage")

    def test_bad_citation_prevents_judge_call(self):
        s = ready_state()
        s["report"]["sections"][0] = ("SUMMARY", "unknown [invented:id]")

        def forbidden(_):
            raise AssertionError("judge must not run")

        result = evaluate(s, chain=RunnableLambda(forbidden))["quality_result"]
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["groundedness"], "fail")

    def test_generation_retains_internal_ids_without_export(self):
        s = ready_state()
        raw = {
            "sections": [
                {"title": t, "content": c} for t, c in s["report"]["sections"]
            ],
            "cited_evidence_ids": ["kivi:a"],
        }
        with patch(
            "kv_cache_eval.features.report.node._write_pdf",
            side_effect=AssertionError("draft only"),
        ):
            result = write_report(s, chain=RunnableLambda(lambda _: raw))
        self.assertEqual(result["report"]["cited_evidence_ids"], ["kivi:a"])

    def test_code_generated_reference_section_may_be_omitted_by_model(self):
        s = ready_state()
        raw = {
            "sections": [
                {"title": t, "content": c} for t, c in s["report"]["sections"][:-1]
            ],
            "cited_evidence_ids": ["kivi:a"],
        }
        result = write_report(s, chain=RunnableLambda(lambda _: raw))
        self.assertEqual(result["report"]["sections"][-1], ("REFERENCE", ""))
        self.assertEqual(result["report"]["cited_evidence_ids"], ["kivi:a"])

    def test_missing_body_section_still_rejected(self):
        s = ready_state()
        raw = {
            "sections": [
                {"title": t, "content": c} for t, c in s["report"]["sections"][1:]
            ],
            "cited_evidence_ids": ["kivi:a"],
        }
        with self.assertRaisesRegex(ValueError, "목차"):
            write_report(s, chain=RunnableLambda(lambda _: raw))

    def test_old_quality_verdict_cannot_export(self):
        s = ready_state()
        s.update(evaluate(s, chain=judge()))
        s["report_revision"] = 1
        with self.assertRaisesRegex(ValueError, "품질 통과"):
            export_pdf(s)

    def test_approved_pdf_is_real_and_contains_korean(self):
        from pypdf import PdfReader

        s = ready_state()
        s.update(evaluate(s, chain=judge()))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                "os.environ", {"REPORT_PDF_PATH": str(Path(directory) / "report.pdf")}
            ),
        ):
            result = export_pdf(s)
            pdf = PdfReader(result["pdf_path"])
            self.assertGreaterEqual(len(pdf.pages), 1)
            self.assertIn("검증용", "".join(page.extract_text() for page in pdf.pages))

    def test_long_pdf_preserves_existing_final(self):
        from pypdf import PdfWriter

        s = ready_state()
        s.update(evaluate(s, chain=judge()))

        def long_pdf(report, path):
            writer = PdfWriter()
            for _ in range(11):
                writer.add_blank_page(width=595, height=842)
            writer.write(str(path))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.pdf"
            output.write_bytes(b"existing")
            with (
                patch.dict("os.environ", {"REPORT_PDF_PATH": str(output)}),
                patch(
                    "kv_cache_eval.features.report.node._write_pdf",
                    side_effect=long_pdf,
                ),
            ):
                with self.assertRaises(ReportTooLong):
                    export_pdf(s)
            self.assertEqual(output.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).iterdir()), [output])
