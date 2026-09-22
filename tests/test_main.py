"""전체 Graph 진입점의 질문 연결을 외부 API 없이 확인한다."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main as app

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.technical_research.node import research_technology


class MainTest(unittest.TestCase):
    def test_import_does_not_execute_graph_or_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {key: value for key, value in os.environ.items()
                           if key not in ("OPENAI_API_KEY", "TAVILY_API_KEY")}
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            completed = subprocess.run([
                os.sys.executable, "-c",
                "import main; from pathlib import Path; assert not Path('data').exists()",
            ], cwd=directory, env=environment, check=True, capture_output=True)
        self.assertEqual(completed.stdout, b"")

    def test_question_reaches_both_graph_research_nodes(self):
        graph = Mock()
        final = {"report": "graph result"}
        graph.invoke.return_value = final

        def research(state, technology, *, user_question):
            self.assertEqual(user_question, app.QUESTION)
            return {"evidence": [{"id": technology.lower()}], "notes": []}

        with patch.object(app, "build_graph", return_value=graph) as build, \
             patch.object(app, "research_technology", side_effect=research) as investigate:
            self.assertIs(app.main(), final)
            overrides = build.call_args.args[0]
            self.assertEqual(set(overrides), {"research_kivi", "research_infinigen"})
            state = graph.invoke.call_args.args[0]
            self.assertEqual(state["selected_technologies"], ("KIVI", "InfiniGen"))
            self.assertEqual(overrides["research_kivi"](state)["kivi_evidence"]["evidence"][0]["id"], "kivi")
            self.assertEqual(overrides["research_infinigen"](state)["infinigen_evidence"]["evidence"][0]["id"], "infinigen")
        self.assertEqual(investigate.call_count, 2)
        graph.invoke.assert_called_once_with(state)

    def test_question_reaches_research_workflow(self):
        with patch("kv_cache_eval.features.technical_research.node.make_runtime_llm", return_value=object()), \
             patch("kv_cache_eval.features.technical_research.workflow.research_questions",
                   return_value={"evidence": [], "notes": []}) as workflow:
            result = research_technology(new_state(), "KIVI", user_question=app.QUESTION)
        questions = workflow.call_args.args[0]
        self.assertEqual(len(questions), 10)
        self.assertTrue(all(app.QUESTION in item.text and "KIVI" in item.text for item in questions))
        self.assertIn("KIVI 조사", result["notes"][-1])

    def test_unimplemented_graph_error_is_not_hidden(self):
        graph = Mock()
        graph.invoke.side_effect = NotImplementedError("보고서 미구현")
        with patch.object(app, "build_graph", return_value=graph):
            with self.assertRaises(NotImplementedError):
                app.main()


if __name__ == "__main__":
    unittest.main()
