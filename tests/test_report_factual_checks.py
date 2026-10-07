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
    factual_issues,
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

        result = evaluate(state, chain=judge())["quality_result"]
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["groundedness"], "fail")
        self.assertIn("PTB", result["issues"][0]["required_action"])
        state.update(quality_result=result)
        with self.assertRaisesRegex(ValueError, "품질 통과"):
            export_pdf(state)

    def test_factual_error_and_other_judge_findings_are_collected_together(self):
        state = ready_state()
        state["report"]["sections"][3] = (
            "3. 평가 기준 및 방법",
            "PTT로 평가했다. [kivi:a]",
        )
        result = evaluate(state, chain=judge("fail"))["quality_result"]
        self.assertEqual(
            {issue["criterion"] for issue in result["issues"]},
            {"groundedness", "perspective_coverage"},
        )

    def test_correct_dataset_and_latency_caveat_are_not_rejected(self):
        state = ready_state()
        state["report"]["sections"][3] = (
            "3. 평가 기준 및 방법",
            "PTB로 평가했다. 처리량 증가만으로 지연 감소를 입증할 수 없다. [kivi:a]",
        )
        self.assertTrue(evaluate(state, chain=judge())["quality_result"]["passed"])

    def test_explicit_dataset_correction_is_not_a_typo(self):
        state = ready_state()
        state["report"]["sections"][3] = (
            "3. 평가 기준 및 방법",
            "데이터셋은 PTB이며, PTT로 읽으면 안 된다. [kivi:a]",
        )
        self.assertEqual(factual_issues(state["report"]), [])
        state["report"]["sections"][3] = (
            "3. 평가 기준 및 방법",
            "PTB가 아니라 PTT로 평가했다. [kivi:a]",
        )
        self.assertTrue(factual_issues(state["report"]))

    def test_infinigen_setup_does_not_disprove_kivi_limitations(self):
        report = {
            "sections": [
                (
                    "SUMMARY",
                    "InfiniGen은 OPT와 Llama-2로 평가했다. KIVI 모델 이름은 미공개다.",
                )
            ]
        }
        evidence = {
            "infinigen:fact-check:infinigen-arxiv-v1:p9": {
                "excerpt": "OPT Llama-2 FlexGen"
            }
        }
        self.assertEqual(factual_issues(report, evidence), [])

    def test_writer_and_judge_receive_the_same_mandatory_rules(self):
        captured = {}

        def factory(schema, prompt, **kwargs):
            captured[kwargs["name"]] = prompt
            if kwargs["name"] == "report_quality":
                return judge()
            return RunnableLambda(
                lambda _: {
                    "cover": {"title": "자동 제목", "subtitle": "", "scope": ""},
                    **{
                        name: [{"text": "검증된 설명", "evidence_ids": ["kivi:a"]}]
                        for name in BODY_FIELDS
                    },
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

    def test_model_disclosure_error_cannot_pass_with_a_passing_judge(self):
        state = ready_state()
        state["report"]["sections"][2] = (
            "2. 기술 선정 및 개요",
            "InfiniGen 원문이 두 모델 이름과 offloading system을 밝히지 않는다. [kivi:a]",
        )
        evidence = collect_report_evidence(state)
        evidence["infinigen:fact-check:infinigen-arxiv-v1:p9"] = {
            "excerpt": "We use OPT and Llama-2 models with UVM and FlexGen."
        }
        with patch(
            "kv_cache_eval.features.quality.node.collect_report_evidence",
            return_value=evidence,
        ):
            result = evaluate(state, chain=judge())["quality_result"]
        self.assertFalse(result["passed"])
        self.assertIn("Llama-2", result["issues"][0]["required_action"])

    def test_excerpt_limit_and_operational_unknown_are_not_non_disclosure(self):
        report = {
            "sections": [
                (
                    "SUMMARY",
                    "InfiniGen은 OPT와 Llama-2로 평가했다. 운영 채택은 미확인이다. 이번 발췌에는 모델 이름이 없다.",
                )
            ]
        }
        evidence = {
            "infinigen:fact-check:infinigen-arxiv-v1:p9": {
                "excerpt": "OPT Llama-2 FlexGen"
            }
        }
        self.assertEqual(factual_issues(report, evidence), [])
        report["sections"] = [
            ("SUMMARY", "InfiniGen 대표 모델 이름의 완전 공개는 제한적이다.")
        ]
        self.assertTrue(factual_issues(report, evidence))
        self.assertEqual(factual_issues(report, {}), [])

    def test_explicit_disclosure_correction_is_not_non_disclosure(self):
        evidence = {
            "infinigen:fact-check:infinigen-arxiv-v1:p9": {
                "excerpt": "OPT Llama-2 FlexGen"
            }
        }
        for text in (
            "InfiniGen 모델 이름은 비공개가 아니다.",
            "InfiniGen 오프로딩 환경은 미공개가 아니라 UVM과 FlexGen으로 명시된다.",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    factual_issues({"sections": [("SUMMARY", text)]}, evidence), []
                )
        self.assertTrue(
            factual_issues(
                {"sections": [("SUMMARY", "InfiniGen 모델 이름은 비공개다.")]},
                evidence,
            )
        )

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
            self.assertEqual(
                set(payload),
                {
                    "question",
                    "domain",
                    "report",
                    "evidence",
                    "validation_feedback",
                    "mandatory_findings",
                },
            )
            self.assertEqual(payload["question"], state["question"])
            self.assertNotIn("evaluations", payload)
            self.assertNotIn("gaps", payload)
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
