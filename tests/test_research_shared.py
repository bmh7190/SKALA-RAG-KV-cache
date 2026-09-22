"""두 기술이 같은 조사 흐름을 쓰면서 출처 귀속과 평가 계약을 지키는지 검사."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.technical_research import node
from kv_cache_eval.features.technical_research.ingest import (
    DEFAULT_DOCUMENT_DIR, load_sources, validate_all_sources,
)
from kv_cache_eval.features.technical_research.prompts import questions_for_state
from kv_cache_eval.features.technical_research.workflow import (
    Claim, Extraction, Review, ground_claim, research_questions,
)


class SharedResearchTest(unittest.TestCase):
    def test_five_pdf_manifests_and_page_budgets(self):
        kivi, kivi_budget = load_sources("KIVI")
        infinigen, infinigen_budget = load_sources("InfiniGen")
        self.assertEqual((kivi_budget, infinigen_budget), (100, 100))
        self.assertEqual(([item.page_count for item in kivi], [item.page_count for item in infinigen]),
                         ([15, 34], [18, 23, 50]))
        self.assertEqual({item.evidence_role for item in kivi}, {"primary", "independent_validation"})
        paths = [DEFAULT_DOCUMENT_DIR / item.file for item in kivi + infinigen]
        if not all(path.is_file() for path in paths):
            self.skipTest("ignored local PDFs are unavailable")
        counts = validate_all_sources()
        self.assertEqual(counts, {"KIVI": 49, "InfiniGen": 91})
        self.assertEqual(sum(counts.values()), 140)

    def test_common_node_uses_same_engine_and_returns_both_keys(self):
        state = new_state()
        def fake_research(current, technology, **kwargs):
            self.assertIs(current, state)
            self.assertEqual(kwargs["user_question"], "공통 질문")
            return {"evidence": [], "notes": [technology]}
        with patch.object(node, "research_technology", side_effect=fake_research) as shared:
            output = node.research(state, user_question="공통 질문")
        self.assertEqual(set(output), {"kivi_evidence", "infinigen_evidence"})
        self.assertEqual(output["kivi_evidence"]["notes"], ["KIVI"])
        self.assertEqual(output["infinigen_evidence"]["notes"], ["InfiniGen"])
        self.assertEqual([call.args[1] for call in shared.call_args_list], ["KIVI", "InfiniGen"])

    def test_research_engine_receives_prior_evidence_on_retry(self):
        state = new_state()
        prior = {"evidence": [], "notes": ["earlier round"]}
        state["kivi_evidence"] = prior
        state["research_round"] = 1
        with patch.object(node, "make_runtime_llm", return_value=object()), \
             patch("kv_cache_eval.features.technical_research.workflow.research_questions",
                   return_value={"evidence": [], "notes": []}) as workflow:
            node.research_technology(state, "KIVI", user_question="같은 질문")
        self.assertIs(workflow.call_args.kwargs["prior"], prior)
        self.assertTrue(all("같은 질문" in item.text for item in workflow.call_args.args[0]))

    def test_both_targets_get_same_general_questions(self):
        state = new_state()
        kivi = questions_for_state(state, "KIVI")
        infinigen = questions_for_state(state, "InfiniGen")
        self.assertEqual([item.id for item in kivi], [item.id for item in infinigen])
        self.assertEqual([item.route for item in kivi], [item.route for item in infinigen])
        self.assertEqual([item.text.replace("KIVI", "TARGET") for item in kivi],
                         [item.text.replace("InfiniGen", "TARGET") for item in infinigen])
        initial = " ".join(item.text for item in kivi)
        for assumed in ("per-channel", "per-token", "KVQuant", "A100", "ShareGPT"):
            self.assertNotIn(assumed, initial)

    def test_independent_target_evaluation_is_attributed_but_background_is_not(self):
        body = "KVQuant compares KIVI on RULER with a stated evaluation setup."
        independent = Document(id="independent", page_content=body, metadata={
            "chunk_id": "independent", "source_id": "kvquant_validation",
            "source_document": "kvquant_validation.pdf", "source_title": "KVQuant paper",
            "source_url": None, "source_role": "independent_validation",
            "subject_technology": "KVQuant", "page": 7,
        })
        claim = Claim(chunk_id="independent", source_id="kvquant_validation", page=7,
                      excerpt="KVQuant compares KIVI on RULER",
                      claim="KVQuant independently compares KIVI on RULER")
        evidence = ground_claim(claim, [independent], target="KIVI", category="independent_evaluation")
        self.assertEqual(evidence["source"]["document"], "kvquant_validation.pdf")
        self.assertIn(":independent_evaluation:", evidence["id"])
        self.assertIsNone(ground_claim(claim.model_copy(update={"claim": "KIVI improves RULER"}),
                                       [independent], target="KIVI", category="independent_evaluation"))
        background = independent.model_copy(update={"metadata": {
            **independent.metadata, "source_role": "comparison_background",
        }})
        self.assertIsNone(ground_claim(claim, [background], target="KIVI", category="independent_evaluation"))

    def test_gap_research_keeps_category_for_maturity_contract(self):
        from kv_cache_eval.features.maturity.node import evaluate

        state = new_state()
        state["research_round"] = 1
        state["evidence_gaps"] = [{"technology": "KIVI", "criterion": "기술 성숙도",
                                   "reason": "핵심 원리 근거가 부족"}]
        gap = questions_for_state(state, "KIVI")[0]
        self.assertEqual(gap.id, "gap-0")
        self.assertEqual(gap.category, "principle")
        sentence = "The key cache should be quantized per-channel and the value cache should be quantized per-token."
        doc = Document(id="kivi-principle", page_content=sentence, metadata={
            "chunk_id": "kivi-principle", "source_id": "kivi_original",
            "source_document": "kivi_original.pdf", "source_title": "KIVI paper",
            "source_url": None, "source_role": "primary", "subject_technology": "KIVI", "page": 1,
        })
        class Reviewer:
            def review(self, *_):
                return Review(sufficient=True, reason="explicit principle")
            def extract(self, *_):
                return Extraction(claims=[Claim(chunk_id="kivi-principle", source_id="kivi_original",
                                                page=1, excerpt=sentence, claim=sentence)])
        result = research_questions([gap], lambda *_: [doc], Reviewer(), target="KIVI")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn(":principle:", result["evidence"][0]["id"])
        maturity = evaluate({**state, "kivi_evidence": result})["maturity_eval"]["evaluations"][0]
        self.assertEqual(maturity["score"], 2.0)

    def test_prior_evidence_never_crosses_technology(self):
        wrong = {"id": "wrong", "technology": "InfiniGen", "claim": "unrelated",
                 "excerpt": "unrelated", "source": {"document": "paper", "url": None, "page": 1},
                 "experiment": None, "limitations": [], "verification_status": "source_checked"}
        class UnusedReviewer:
            def review(self, *_):
                raise AssertionError("no questions")
            def extract(self, *_):
                raise AssertionError("no questions")
        result = research_questions([], lambda *_: [], UnusedReviewer(), target="KIVI",
                                    prior={"evidence": [wrong], "notes": ["wrong prior"]})
        self.assertEqual(result, {"evidence": [], "notes": []})


if __name__ == "__main__":
    unittest.main()
