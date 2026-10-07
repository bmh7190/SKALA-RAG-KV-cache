import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_report_quality import ready_state

from kv_cache_eval.features.report.source_audit import collect_report_evidence
from kv_cache_eval.features.technical_research.ingest import Source


class SourceAuditTests(unittest.TestCase):
    def source(self):
        return Source(
            "primary",
            "paper",
            "Source title",
            "https://example.org/paper",
            "v1",
            "primary",
            "KIVI",
            "paper.pdf",
            3,
            "0" * 64,
        )

    def test_direct_throughput_survives_retrieval_gap_and_reference_pages_stop_audit(
        self,
    ):
        state = ready_state()
        state["selected_technologies"] = ("KIVI",)
        state["evidence_gaps"] = [
            {"technology": "KIVI", "criterion": "처리량", "reason": "미확인"}
        ]
        pages = [
            SimpleNamespace(
                extract_text=lambda: "Performance: throughput 27.36 tokens per second"
            ),
            SimpleNamespace(
                extract_text=lambda: "References\nOther study 999 tokens per second"
            ),
            SimpleNamespace(extract_text=lambda: "Appendix"),
        ]
        with (
            patch(
                "kv_cache_eval.features.report.source_audit.load_sources",
                return_value=([self.source()], 100),
            ),
            patch(
                "kv_cache_eval.features.report.source_audit.validate_sources"
            ) as validate,
            patch("pypdf.PdfReader", return_value=SimpleNamespace(pages=pages)),
        ):
            evidence = collect_report_evidence(state)
        validate.assert_called_once()
        item = evidence["kivi:source-audit:primary:p1"]
        self.assertIn("27.36", item["excerpt"])
        self.assertEqual(item["source"]["page"], 1)
        self.assertNotIn("kivi:source-audit:primary:p2", evidence)
        self.assertEqual(state["evidence_gaps"][0]["reason"], "미확인")

    def test_changed_pdf_is_rejected_before_auditing(self):
        state = ready_state()
        state["selected_technologies"] = ("KIVI",)
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "paper.pdf").write_bytes(b"not the original PDF")
            with patch(
                "kv_cache_eval.features.report.source_audit.load_sources",
                return_value=([self.source()], 100),
            ):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    collect_report_evidence(state, Path(folder))
