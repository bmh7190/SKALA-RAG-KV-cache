"""실제 근거의 잘못된 인용과 무한 재검색을 막는 오프라인 검증."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


RAG_AVAILABLE = importlib.util.find_spec("numpy") is not None
FIXTURE_SHA256 = "06df8c0f354d22d027537ae66c6ed91957c1ca732b3bffb44c1df43614327db8"


@unittest.skipUnless(RAG_AVAILABLE, "rag extra 설치 후 실행")
class InfiniGenTest(unittest.TestCase):
    def hit(self, role="primary"):
        from langchain_core.documents import Document

        subject = "InfiniGen" if role == "primary" else "FlexGen"
        return Document(id="chunk-one", page_content="InfiniGen keeps a KV pool in CPU memory.", metadata={
            "chunk_id": "chunk-one",
            "source_id": "infinigen-arxiv-v1" if role == "primary" else "flexgen-icml-2023",
            "source_title": "Paper", "source_url": "https://example.org/paper.pdf",
            "source_role": role, "subject_technology": subject, "page": 6,
        })

    def test_grounding_rejects_wrong_source_page_excerpt_and_background_attribution(self):
        from kv_cache_eval.features.technical_research.infinigen.workflow import Claim, ground_claim

        hit = self.hit()
        valid = Claim(chunk_id=hit.metadata["chunk_id"], source_id=hit.metadata["source_id"], page=6,
                      excerpt="KV pool in CPU memory", claim="InfiniGen keeps a KV pool in CPU memory")
        evidence = ground_claim(valid, [hit], target="InfiniGen")
        self.assertEqual(evidence["verification_status"], "source_checked")
        self.assertEqual(evidence["source"]["page"], 6)
        self.assertIsNone(ground_claim(valid, [hit], target="OtherTech"))
        self.assertIsNone(ground_claim(valid.model_copy(update={"page": 5}), [hit], target="InfiniGen"))
        self.assertIsNone(ground_claim(valid.model_copy(update={"source_id": "other"}), [hit], target="InfiniGen"))
        self.assertIsNone(ground_claim(valid.model_copy(update={"excerpt": "not in the page"}), [hit], target="InfiniGen"))
        background = self.hit("system_background")
        wrong = valid.model_copy(update={"source_id": background.metadata["source_id"]})
        self.assertIsNone(ground_claim(wrong, [background], target="InfiniGen"))

    def test_experiment_context_requires_same_source_chunk(self):
        from langchain_core.documents import Document
        from kv_cache_eval.features.technical_research.infinigen.workflow import Claim, ground_claim

        text = "OPT-6.7B long sequence decoding is compared with FlexGen."
        hit = Document(id="experiment-chunk", page_content=text, metadata={
            "chunk_id": "experiment-chunk", "source_id": "infinigen-arxiv-v1",
            "source_title": "Paper", "source_url": "https://example.org/paper.pdf",
            "source_role": "primary", "subject_technology": "InfiniGen", "page": 9,
        })
        claim = Claim(chunk_id="experiment-chunk", source_id="infinigen-arxiv-v1", page=9,
                      excerpt="long sequence decoding", claim="The paper compares long sequence decoding.",
                      model="OPT-6.7B", workload="GPU 기반 클라우드 LLM 서비스", baseline="FlexGen")
        evidence = ground_claim(claim, [hit], target="InfiniGen")
        self.assertEqual(evidence["experiment"], {"model": "OPT-6.7B", "baseline": "FlexGen"})

    def test_initial_questions_use_only_target_and_common_goals(self):
        from kv_cache_eval.common.state import new_state
        from kv_cache_eval.features.technical_research.infinigen.prompts import (
            EXTRACT_SYSTEM, REVIEW_SYSTEM, questions_for_state,
        )

        questions = questions_for_state(new_state(), target="NewTech")
        self.assertEqual(len(questions), 6)
        self.assertTrue(all("NewTech" in item.text for item in questions))
        text = " ".join(item.text for item in questions) + REVIEW_SYSTEM + EXTRACT_SYSTEM
        for preset in ("InfiniGen", "KV pool", "partial weight", "partial key", "PCIe", "FlexGen", "H2O"):
            self.assertNotIn(preset, text)

    def test_requery_preserves_goal_and_uses_only_observed_terms(self):
        from langchain_core.documents import Document
        from kv_cache_eval.features.technical_research.infinigen.workflow import Review, build_question_graph

        passage = "The observed transfer pattern reduces resource use."
        hit = Document(id="search-chunk", page_content=passage, metadata={
            "chunk_id": "search-chunk", "source_id": "newtech-paper",
            "source_title": "NewTech paper", "source_url": "https://example.org/paper.pdf",
            "source_role": "primary", "subject_technology": "NewTech", "page": 2,
        })
        queries = []
        def search(query, _):
            queries.append(query)
            return [hit]
        def review(*_):
            return Review(sufficient=False, reason="more detail needed",
                          refinement_terms=["observed transfer pattern", "invented accelerator"])
        graph = build_question_graph(search, review, lambda *_: self.fail("no extraction"), target="NewTech")
        goal = "NewTech의 자원 절감 근거는 무엇인가?"
        graph.invoke({"question": goal, "query": "", "attempt": 0, "max_attempts": 2,
                      "top_k": 3, "hits": [], "review": None, "evidence": [], "notes": []})
        self.assertEqual(len(queries), 2)
        self.assertTrue(queries[1].startswith(goal))
        self.assertIn("observed transfer pattern", queries[1])
        self.assertNotIn("invented accelerator", queries[1])

    def test_retry_is_bounded_and_unresolved(self):
        from kv_cache_eval.features.technical_research.infinigen.workflow import build_question_graph

        queries = []
        def search(query, top_k):
            queries.append(query)
            return []
        graph = build_question_graph(search, lambda *_: self.fail("no hits"),
                                     lambda *_: self.fail("no extraction"), target="InfiniGen")
        result = graph.invoke({"question": "missing fact", "query": "", "attempt": 0,
                               "max_attempts": 2, "top_k": 3, "hits": [], "review": None,
                               "evidence": [], "notes": []})
        self.assertEqual(len(queries), 2)
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any("미확인" in note for note in result["notes"]))

    def test_call_budget_records_current_and_remaining_questions(self):
        from kv_cache_eval.features.technical_research.infinigen.prompts import ResearchQuestion
        from kv_cache_eval.features.technical_research.infinigen.workflow import CallBudgetExhausted, research_questions

        class FakeRetriever:
            def search(self, *_):
                return [self_hit]
        class FakeReviewer:
            def review(self, *_):
                raise CallBudgetExhausted("limit")
            def extract(self, *_):
                raise AssertionError("call budget should stop before extraction")
        self_hit = self.hit()
        result = research_questions([ResearchQuestion("q1", "first"), ResearchQuestion("q2", "second")],
                                    FakeRetriever(), FakeReviewer(), target="InfiniGen")
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any("q1: first" in note for note in result["notes"]))
        self.assertTrue(any("q2: second" in note for note in result["notes"]))

    def test_partial_state_update(self):
        from kv_cache_eval.common.state import new_state
        from kv_cache_eval.features.technical_research.infinigen import node

        result = {"evidence": [], "notes": ["fixture only"]}
        with patch.object(node, "make_runtime_llm", return_value=object()), \
             patch.object(node, "ensure_index", return_value=(object(), {})), \
             patch.object(node, "ModelReviewer") as reviewer, \
             patch.object(node, "research_questions", return_value=result):
            reviewer.return_value.calls = 0
            output = node.research_infinigen(new_state())
        self.assertEqual(set(output), {"infinigen_evidence"})
        self.assertEqual(output["infinigen_evidence"]["evidence"], [])

    def test_fingerprint_includes_citation_metadata_and_checksum(self):
        from kv_cache_eval.features.technical_research.infinigen.ingest import Source, file_sha256
        from kv_cache_eval.features.technical_research.infinigen.retriever import (
            INDEX_FORMAT, RetrievalSettings, index_fingerprint, index_is_current,
        )

        source = Source("id", "title", "https://example.org", "v1", "primary",
                        "InfiniGen", "paper.pdf", 1, "abc")
        settings = RetrievalSettings("BAAI/bge-m3")
        fingerprint = index_fingerprint([source], settings)
        self.assertNotEqual(fingerprint, index_fingerprint([replace(source, source_url="https://changed.org")], settings))
        self.assertNotEqual(fingerprint, index_fingerprint([replace(source, evidence_role="system_background")], settings))
        self.assertNotEqual(fingerprint, index_fingerprint([source], replace(settings, chunk_tokens=256)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "index.faiss").write_bytes(b"index")
            (path / "index.pkl").write_bytes(b"local cache")
            (path / "manifest.json").write_text(json.dumps({
                "format": INDEX_FORMAT, "fingerprint": fingerprint,
                "created_by": "kv-cache-eval:infinigen",
                "files": {"index.faiss": file_sha256(path / "index.faiss"),
                          "index.pkl": file_sha256(path / "index.pkl")},
            }))
            self.assertTrue(index_is_current(path, fingerprint))
            (path / "index.pkl").write_text("changed")
            self.assertFalse(index_is_current(path, fingerprint))

    def test_fixed_fixture_and_source_pages(self):
        from kv_cache_eval.features.technical_research.infinigen.ingest import (
            DEFAULT_DOCUMENT_DIR, load_sources, validate_sources,
        )

        path = Path("tests/fixtures/infinigen_questions.json")
        payload = path.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), FIXTURE_SHA256)
        questions = json.loads(payload)["questions"]
        self.assertEqual(sum(item["answerable"] for item in questions), 12)
        self.assertEqual(sum(not item["answerable"] for item in questions), 2)
        sources, budget = load_sources()
        known_pages = {source.document_id: source.page_count for source in sources}
        for question in questions:
            for expected in question["relevant_sources"]:
                self.assertTrue(all(1 <= page <= known_pages[expected["document_id"]]
                                    for page in expected["pdf_pages"]))
        if DEFAULT_DOCUMENT_DIR.exists() and list(DEFAULT_DOCUMENT_DIR.glob("*.pdf")):
            self.assertEqual(validate_sources(sources, DEFAULT_DOCUMENT_DIR, budget), 91)


if __name__ == "__main__":
    unittest.main()
