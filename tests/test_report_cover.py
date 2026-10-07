import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader
from test_report_quality import judge, ready_state

from kv_cache_eval.features.quality.node import evaluate
from kv_cache_eval.features.report.node import export_pdf
from kv_cache_eval.features.report.pdf import _write_pdf


class ReportCoverTests(unittest.TestCase):
    def test_long_generated_cover_fits_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            _write_pdf(
                {
                    "cover": {
                        "title": "가" * 80,
                        "subtitle": "나" * 120,
                        "scope": "다" * 150,
                    },
                    "sections": [("SUMMARY", "검증용 본문")],
                },
                path,
            )
            reader = PdfReader(path)
            self.assertEqual(len(reader.pages), 2)
            text = "".join(reader.pages[0].extract_text().split())
            self.assertIn("가" * 80, text)
            self.assertIn("나" * 120, text)
            self.assertIn("다" * 150, text)

    def test_generated_cover_schema_requires_valid_title(self):
        from pydantic import ValidationError

        from kv_cache_eval.features.report.cover import CoverOutput

        for title in ("", "   ", "가" * 81, "제목 [근거]", "제목\n다음 줄"):
            with self.subTest(title=title), self.assertRaises(ValidationError):
                CoverOutput(title=title, subtitle="", scope="")

    def test_cover_survives_citation_numbering_and_legacy_subject_is_dynamic(self):
        from kv_cache_eval.common.evidence import collect_evidence
        from kv_cache_eval.features.report.cover import cover_from_state
        from kv_cache_eval.features.report.node import printable_report

        state = ready_state()
        state["report"]["cover"] = {
            "title": "이번 실행의 제목",
            "subtitle": "",
            "scope": "",
        }
        printable = printable_report(state["report"], collect_evidence(state))
        self.assertEqual(printable["cover"], state["report"]["cover"])
        state["selected_technologies"] = ("테스트 기술 A", "테스트 기술 B")
        state["domain_and_criteria"]["domain"] = "다른 적용 영역"
        generated = cover_from_state(state)
        self.assertIn("테스트 기술 A", generated.title)
        self.assertEqual(generated.subtitle, "다른 적용 영역")
        self.assertNotIn("KIVI", generated.title)

    def test_cover_is_separate_and_body_starts_at_page_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            with patch.dict(
                "os.environ",
                {
                    "REPORT_AUTHORS": "홍길동 <연구> & 김연구",
                    "REPORT_AFFILIATION": "SKALA",
                    "REPORT_DATE": "2026-10-07",
                },
            ):
                _write_pdf(
                    {
                        "cover": {
                            "title": "사용자 질문 기반 평가",
                            "subtitle": "동적으로 생성한 비교 대상",
                            "scope": "실제 평가 영역",
                        },
                        "sections": [("SUMMARY", "본문 검증용 내용 [1]")],
                    },
                    path,
                )
            reader = PdfReader(path)
            self.assertEqual(len(reader.pages), 2)
            cover, body = [page.extract_text() for page in reader.pages]
            self.assertIn("사용자 질문 기반 평가", cover)
            self.assertIn("동적으로 생성한 비교 대상", cover)
            self.assertNotIn("KIVI", cover)
            self.assertEqual(reader.metadata.title, "사용자 질문 기반 평가")
            self.assertIn("홍길동 <연구> & 김연구", cover)
            self.assertIn("2026-10-07", cover)
            self.assertNotIn("SUMMARY", cover)
            self.assertIn("SUMMARY", body)
            self.assertIn("본문 검증용 내용 [1]", body)
            self.assertEqual(body.strip().splitlines()[0], "1")
            self.assertEqual(reader.metadata.author, "홍길동 <연구> & 김연구")

    def test_cover_does_not_invent_authors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            with patch.dict("os.environ", {"REPORT_AUTHORS": "", "REPORT_DATE": ""}):
                _write_pdf({"sections": [("SUMMARY", "검증용 본문")]}, path)
            cover = PdfReader(path).pages[0].extract_text()
            self.assertNotIn("작성자", cover)
            self.assertIn("작성일", cover)

    def test_invalid_cover_preserves_existing_final_pdf(self):
        state = ready_state()
        state.update(evaluate(state, chain=judge()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            path.write_bytes(b"previous approved report")
            with patch.dict(
                "os.environ",
                {
                    "REPORT_PDF_PATH": str(path),
                    "REPORT_AUTHORS": "가" * 501,
                },
            ):
                with self.assertRaisesRegex(ValueError, "500자"):
                    export_pdf(state)
            self.assertEqual(path.read_bytes(), b"previous approved report")
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
