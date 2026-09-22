"""main 진입점의 질문 전달과 중간 결과 보존을 API 없이 확인한다."""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as app

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.technical_research.node import research_technology
from kv_cache_eval.features.technical_research.prompts import questions_for_state


class MainTest(unittest.TestCase):
    def test_import_does_not_run_or_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {key: value for key, value in os.environ.items()
                           if key not in ("OPENAI_API_KEY", "TAVILY_API_KEY")}
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            subprocess.run([
                os.sys.executable, "-c",
                "import main; from pathlib import Path; assert not Path('data/cache/runs').exists()",
            ], cwd=directory, env=environment, check=True, capture_output=True)

    def test_question_reaches_both_research_results(self):
        state = new_state()
        for technology in state["selected_technologies"]:
            base = questions_for_state(state, technology)
            with_request = questions_for_state(state, technology, app.QUESTION)
            self.assertEqual([(item.id, item.route, item.category) for item in base],
                             [(item.id, item.route, item.category) for item in with_request])
            self.assertTrue(all(app.QUESTION in item.text and technology in item.text
                                for item in with_request))

        calls = []

        def research(state, technology, *, user_question):
            calls.append((technology, user_question))
            if technology == "InfiniGen":
                self.assertIsNotNone(state["kivi_evidence"])
            return {"evidence": [{"id": technology.lower()}], "notes": []}

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, "RUNS_DIR", Path(directory)), \
             patch.object(app, "research_technology", side_effect=research), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(app.main(), 0)
            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(files), 1)
            saved = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(calls, [("KIVI", app.QUESTION), ("InfiniGen", app.QUESTION)])
        self.assertEqual(saved["question"], app.QUESTION)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["completed_technologies"], ["KIVI", "InfiniGen"])
        self.assertEqual(saved["state"]["kivi_evidence"]["evidence"][0]["id"], "kivi")
        self.assertEqual(saved["state"]["infinigen_evidence"]["evidence"][0]["id"], "infinigen")
        self.assertIn("KIVI 조사 완료", output.getvalue())
        self.assertIn("InfiniGen 조사 완료", output.getvalue())

    def test_question_reaches_research_workflow(self):
        with patch("kv_cache_eval.features.technical_research.node.make_runtime_llm", return_value=object()), \
             patch("kv_cache_eval.features.technical_research.workflow.research_questions",
                   return_value={"evidence": [], "notes": []}) as workflow:
            result = research_technology(new_state(), "KIVI", user_question=app.QUESTION)
        questions = workflow.call_args.args[0]
        self.assertEqual(len(questions), 10)
        self.assertTrue(all(app.QUESTION in item.text and "KIVI" in item.text for item in questions))
        self.assertIn("KIVI 조사", result["notes"][-1])

    def test_first_result_survives_second_failure(self):
        def research(state, technology, *, user_question):
            if technology == "InfiniGen":
                raise RuntimeError("test failure")
            return {"evidence": [{"id": "kivi"}], "notes": []}

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, "RUNS_DIR", Path(directory)), \
             patch.object(app, "research_technology", side_effect=research), \
             contextlib.redirect_stdout(io.StringIO()) as output, \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(app.main(), 1)
            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(files), 1)
            saved = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["failed_technology"], "InfiniGen")
        self.assertEqual(saved["completed_technologies"], ["KIVI"])
        self.assertEqual(saved["state"]["kivi_evidence"]["evidence"][0]["id"], "kivi")
        self.assertIsNone(saved["state"]["infinigen_evidence"])
        self.assertNotIn("두 기술 조사 결과", output.getvalue())
        self.assertIn("InfiniGen 조사 실패", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
