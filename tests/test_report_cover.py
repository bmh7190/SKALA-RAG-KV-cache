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
                _write_pdf({"sections": [("SUMMARY", "본문 검증용 내용 [1]")]}, path)
            reader = PdfReader(path)
            self.assertEqual(len(reader.pages), 2)
            cover, body = [page.extract_text() for page in reader.pages]
            self.assertIn("KIVI와 InfiniGen 비교", cover)
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
