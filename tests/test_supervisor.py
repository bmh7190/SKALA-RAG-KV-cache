import tempfile
import unittest
from pathlib import Path

from supervisor_fixtures import evidence, nodes

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.supervisor.node import supervise
from kv_cache_eval.graph.workflow import build_graph, run


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.events = []
        self.nodes = nodes(self.tmp.name, self.events)

    def invoke(self, **limits):
        return build_graph(self.nodes).invoke(
            new_state(**limits), {"recursion_limit": 100}
        )

    def test_full_flow_only_exports_after_quality(self):
        s = self.invoke()
        self.assertEqual(s["status"], "completed")
        self.assertEqual(self.events[0], "technical_research")
        self.assertEqual(
            self.events[-4:], ["synthesis", "report", "quality", "export_pdf"]
        )
        self.assertTrue(Path(s["pdf_path"]).is_file())
        self.assertEqual(s["step_count"], len(self.events))

    def test_missing_perspective_changes_next_agent(self):
        s = self.invoke()
        s.update(status="running", next_agent=None, last_result=None)
        s["completed_agents"].remove("market")
        decision = supervise(s)
        self.assertEqual(decision["next_agent"], "market")

    def test_targeted_research_then_reevaluate_without_repeating_market(self):
        domain = self.nodes["domain"]
        research = self.nodes["technical_research"]
        requests = []

        def incomplete(s):
            update = domain(s)
            if not s["gap_attempts"]:
                for row in update["domain_eval"]["evaluations"]:
                    if (
                        row["technology"] == "InfiniGen"
                        and row["criterion"] == "처리량"
                    ):
                        row.update(
                            judgment=None,
                            score=None,
                            evidence_ids=[],
                            basis_status="unverified",
                        )
            return update

        def targeted(s):
            requests.append(s["retry_request"])
            result = research(s)
            if s["retry_request"]:
                result["infinigen_evidence"]["evidence"].append(
                    evidence("InfiniGen", "extra")
                )
            return result

        self.nodes.update(domain=incomplete, technical_research=targeted)
        s = self.invoke()
        self.assertEqual(s["status"], "completed")
        self.assertEqual(requests[1]["technology"], "InfiniGen")
        self.assertIn("performance_results", requests[1]["criteria"])
        self.assertEqual(self.events.count("market"), 1)
        self.assertEqual(s["kivi_evidence"]["evidence"][0]["id"], "KIVI:base")

    def test_quality_failure_revises_without_research(self):
        quality = self.nodes["quality"]

        def reject_first(s):
            update = quality(s)
            if s["report_revision"] == 0:
                q = update["quality_result"]
                q["passed"] = False
                q["checks"]["perspective_coverage"] = "fail"
                q["issues"] = [
                    {
                        "criterion": "perspective_coverage",
                        "section": "4. 관점별 평가 결과",
                        "reason": "설명 누락",
                        "required_action": "기존 평가 반영",
                    }
                ]
            return update

        self.nodes["quality"] = reject_first
        s = self.invoke()
        self.assertEqual(s["status"], "completed")
        self.assertEqual(s["report_revision"], 1)
        self.assertEqual(self.events.count("technical_research"), 1)
        self.assertEqual(self.events.count("report"), 2)

    def test_failed_generation_never_exports_previous_pass(self):
        def broken(s):
            raise ValueError("bad citation")

        self.nodes["report"] = broken
        s = self.invoke(max_report_revisions=1)
        self.assertEqual(s["status"], "incomplete")
        self.assertNotIn("export_pdf", self.events)
        self.assertNotIn("quality", self.events)

    def test_timeout_retries_but_authentication_does_not(self):
        original = self.nodes["market"]
        count = []

        def flaky(s):
            count.append(1)
            if len(count) == 1:
                raise TimeoutError("temporary")
            return original(s)

        self.nodes["market"] = flaky
        s = self.invoke()
        self.assertEqual(s["status"], "completed")
        self.assertEqual(len(count), 2)

        class AuthenticationError(Exception):
            pass

        def auth(s):
            raise AuthenticationError("secret must not be logged")

        self.nodes["market"] = auth
        s = self.invoke()
        self.assertEqual(s["status"], "failed")
        self.assertEqual(s["agent_calls"]["market"], 1)
        self.assertNotIn("secret", s["last_error"]["message"])

    def test_budget_exhaustion_is_not_success(self):
        s = self.invoke(max_steps=2)
        self.assertEqual(s["status"], "incomplete")
        self.assertIsNone(s["pdf_path"])
        self.assertEqual(s["step_count"], 2)

    def test_out_of_scope_state_update_rejected(self):
        self.nodes["domain"] = lambda s: {"status": "completed"}
        s = self.invoke()
        self.assertEqual(s["status"], "failed")
        self.assertIsNone(s["pdf_path"])

    def test_missing_evaluation_is_failure_not_evidence_gap(self):
        self.nodes["market"] = lambda s: {
            "market_evidence": {"evidence": [], "notes": []}
        }
        s = self.invoke()
        self.assertEqual(s["status"], "failed")
        self.assertEqual(s["last_error"]["code"], "invalid_result")
        self.assertEqual(s["agent_calls"]["market"], 1)

    def test_sqlite_resume_does_not_repeat_finished_research(self):
        checkpoint = Path(self.tmp.name) / "checkpoints.sqlite"
        s = run(
            "question",
            run_id="resume-test",
            checkpoint_path=checkpoint,
            overrides=self.nodes,
            interrupt_after=["technical_research"],
        )
        self.assertEqual(s["status"], "running")
        self.assertEqual(self.events, ["technical_research"])
        final = run(
            "ignored on resume",
            run_id="resume-test",
            resume=True,
            checkpoint_path=checkpoint,
            overrides=self.nodes,
        )
        self.assertEqual(final["status"], "completed")
        self.assertEqual(self.events.count("technical_research"), 1)
        self.assertEqual(final["question"], "question")
        self.assertTrue(
            (checkpoint.parent / "resume-test" / "decisions.jsonl").is_file()
        )
        with self.assertRaisesRegex(ValueError, "존재"):
            run(
                "question",
                run_id="resume-test",
                checkpoint_path=checkpoint,
                overrides=self.nodes,
            )

    def test_stale_quality_result_cannot_authorize_export(self):
        original = self.nodes["quality"]

        def stale(s):
            update = original(s)
            update["quality_result"]["report_revision"] = -1
            return update

        self.nodes["quality"] = stale
        s = self.invoke(max_agent_calls=2)
        self.assertNotEqual(s["status"], "completed")
        self.assertNotIn("export_pdf", self.events)
