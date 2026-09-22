"""실제 근거의 잘못된 인용과 무한 재검색을 막는 오프라인 검증."""

import hashlib
import importlib.util
import json
import os
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
        self.assertEqual(len(questions), 10)
        self.assertEqual([item.route for item in questions[-2:]], ["web", "both"])
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
        routes = []
        def search(query, _, route):
            queries.append(query)
            routes.append(route)
            return [hit], []
        def review(*_):
            return Review(sufficient=False, reason="more detail needed",
                          refinement_terms=["observed transfer pattern", "invented accelerator"])
        graph = build_question_graph(search, review, lambda *_: self.fail("no extraction"), target="NewTech")
        goal = "NewTech의 자원 절감 근거는 무엇인가?"
        graph.invoke({"question": goal, "route": "rag", "query": "", "attempt": 0, "max_attempts": 2,
                      "top_k": 3, "hits": [], "review": None, "evidence": [], "notes": []})
        self.assertEqual(len(queries), 2)
        self.assertTrue(queries[1].startswith(goal))
        self.assertIn("observed transfer pattern", queries[1])
        self.assertNotIn("invented accelerator", queries[1])
        self.assertEqual(routes, ["rag", "both"])

    def test_retry_is_bounded_and_unresolved(self):
        from kv_cache_eval.features.technical_research.infinigen.workflow import build_question_graph

        queries = []
        def search(query, top_k, route):
            queries.append(query)
            return [], []
        graph = build_question_graph(search, lambda *_: self.fail("no hits"),
                                     lambda *_: self.fail("no extraction"), target="InfiniGen")
        result = graph.invoke({"question": "missing fact", "route": "rag", "query": "", "attempt": 0,
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
                                    FakeRetriever().search, FakeReviewer(), target="InfiniGen")
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any("q1: first" in note for note in result["notes"]))
        self.assertTrue(any("q2: second" in note for note in result["notes"]))

    def test_partial_state_update(self):
        from kv_cache_eval.common.state import new_state
        from kv_cache_eval.features.technical_research import node

        result = {"evidence": [], "notes": ["fixture only"]}
        with patch.object(node, "make_runtime_llm", return_value=object()), \
             patch("kv_cache_eval.features.technical_research.workflow.ModelReviewer") as reviewer, \
             patch("kv_cache_eval.features.technical_research.workflow.research_questions", return_value=result):
            reviewer.return_value.calls = 0
            output = node.research_infinigen(new_state())
        self.assertEqual(set(output), {"infinigen_evidence"})
        self.assertEqual(output["infinigen_evidence"]["evidence"], [])

    def test_fingerprint_includes_citation_metadata_and_checksum(self):
        from kv_cache_eval.features.technical_research.infinigen.ingest import Source, file_sha256
        from kv_cache_eval.features.technical_research.infinigen.retriever import (
            INDEX_FORMAT, RetrievalSettings, index_fingerprint, index_is_current,
        )

        source = Source("id", "id", "title", "https://example.org", "v1", "primary",
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
                "created_by": "kv-cache-eval:technical-research",
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

    def test_web_uses_extracted_body_and_tool_url_only(self):
        from kv_cache_eval.features.technical_research.infinigen.web import search_web

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}), \
             patch("langchain_tavily.TavilySearch") as search, \
             patch("langchain_tavily.TavilyExtract") as extract:
            search.return_value.invoke.return_value = {"results": [
                {"title": "Project page", "url": "https://example.org/project", "content": "search snippet only"},
                {"title": "Unavailable page", "url": "https://example.org/missing", "content": "summary"},
            ]}
            extract.return_value.invoke.return_value = {"results": [
                {"url": "https://example.org/project", "raw_content": "The project page lists a public implementation."},
                {"url": "https://unlisted.example/page", "raw_content": "must be ignored"},
            ], "failed_results": [{"url": "https://example.org/missing"}]}
            docs, notes = search_web("InfiniGen public implementation", "InfiniGen")
        self.assertTrue(any("일부 추출 실패" in note for note in notes))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["page"], None)
        self.assertEqual(docs[0].metadata["source_url"], "https://example.org/project")
        self.assertEqual(docs[0].metadata["source_role"], "web_external")
        self.assertNotIn("search snippet", docs[0].page_content)

    def test_web_grounding_requires_null_page_url_and_excerpt(self):
        from langchain_core.documents import Document
        from kv_cache_eval.features.technical_research.infinigen.workflow import Claim, ground_claim

        doc = Document(id="web-one", page_content="The project page lists a public implementation.", metadata={
            "chunk_id": "web-one", "source_id": "web-id", "source_title": "Project page",
            "source_url": "https://example.org/project", "source_kind": "web",
            "source_role": "web_external", "subject_technology": "InfiniGen", "page": None,
        })
        claim = Claim(chunk_id="web-one", source_id="web-id", page=None,
                      excerpt="lists a public implementation", claim="The project page lists a public implementation.")
        evidence = ground_claim(claim, [doc], target="InfiniGen")
        self.assertEqual(evidence["source"], {"document": "Project page",
                                              "url": "https://example.org/project", "page": None})
        self.assertEqual(evidence["verification_status"], "source_checked")
        self.assertTrue(any("공식성" in item for item in evidence["limitations"]))
        self.assertIsNone(ground_claim(claim.model_copy(update={"page": 1}), [doc], target="InfiniGen"))
        self.assertIsNone(ground_claim(claim.model_copy(update={"excerpt": "search snippet only"}), [doc], target="InfiniGen"))
        bad_url = doc.model_copy(update={"metadata": {**doc.metadata, "source_url": "invented-url"}})
        self.assertIsNone(ground_claim(claim, [bad_url], target="InfiniGen"))

    def test_web_tool_failures_are_notes_not_absence_claims(self):
        from kv_cache_eval.features.technical_research.infinigen.web import search_web

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}), \
             patch("langchain_tavily.TavilySearch") as search, \
             patch("langchain_tavily.TavilyExtract") as extract:
            search.return_value.invoke.side_effect = RuntimeError("private provider details")
            docs, notes = search_web("public support", "InfiniGen")
            self.assertEqual(docs, [])
            self.assertTrue(any("웹 검색 실패" in note and "근거 부재로 판단하지 않음" in note for note in notes))
            self.assertFalse(any("private provider details" in note for note in notes))

            search.return_value.invoke.side_effect = None
            search.return_value.invoke.return_value = {"results": [
                {"title": "Page", "url": "https://example.org/page"},
            ]}
            extract.return_value.invoke.side_effect = RuntimeError("private extraction details")
            docs, notes = search_web("public support", "InfiniGen")
            self.assertEqual(docs, [])
            self.assertTrue(any("웹 본문 추출 실패" in note and "근거 부재로 판단하지 않음" in note for note in notes))
            self.assertFalse(any("private extraction details" in note for note in notes))

    def test_web_only_skips_rag_and_missing_key_is_unresolved(self):
        from kv_cache_eval.features.technical_research.infinigen.prompts import ResearchQuestion
        from kv_cache_eval.features.technical_research.infinigen.web import search_web
        from kv_cache_eval.features.technical_research.infinigen.workflow import research_questions

        calls = []
        class Reviewer:
            def review(self, *_):
                calls.append("review")
                raise AssertionError("no body to review")
            def extract(self, *_):
                raise AssertionError("no body to extract")
        def rag_search(*_):
            raise AssertionError("web-only question loaded RAG")
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            result = research_questions([ResearchQuestion("public", "current support", "web")],
                                        rag_search, Reviewer(), target="InfiniGen", max_attempts=2,
                                        web_search=lambda query, count: search_web(query, "InfiniGen", count))
        self.assertEqual(calls, [])
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any("TAVILY_API_KEY 없음" in note for note in result["notes"]))
        self.assertTrue(any("미확인" in note for note in result["notes"]))

    def test_web_only_extracts_attributed_source_checked_evidence(self):
        from langchain_core.documents import Document
        from kv_cache_eval.features.technical_research.infinigen.prompts import ResearchQuestion
        from kv_cache_eval.features.technical_research.infinigen.workflow import (
            Claim, Extraction, Review, research_questions,
        )

        doc = Document(id="web-one", page_content="The public project page lists an implementation.", metadata={
            "chunk_id": "web-one", "source_id": "web-project", "source_title": "Project page",
            "source_url": "https://example.org/project", "source_kind": "web",
            "source_role": "web_external", "subject_technology": "InfiniGen", "page": None,
        })
        class Reviewer:
            def review(self, *_):
                return Review(sufficient=True, reason="extracted body supports attributed statement")
            def extract(self, *_):
                return Extraction(claims=[Claim(chunk_id="web-one", source_id="web-project", page=None,
                                                excerpt="lists an implementation",
                                                claim="The Project page lists an implementation.")])
        def rag_search(*_):
            raise AssertionError("web-only question loaded RAG")
        result = research_questions([ResearchQuestion("public", "current implementation", "web")],
                                    rag_search, Reviewer(), target="InfiniGen",
                                    web_search=lambda *_: ([doc], []))
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["source"]["page"], None)
        self.assertEqual(result["evidence"][0]["verification_status"], "source_checked")

    def test_both_keeps_pdf_evidence_on_web_failure(self):
        from kv_cache_eval.features.technical_research.infinigen.prompts import ResearchQuestion
        from kv_cache_eval.features.technical_research.infinigen.workflow import (
            Claim, Extraction, Review, research_questions,
        )

        pdf = self.hit()
        class Reviewer:
            def review(self, question, hits):
                return Review(sufficient=True, reason="PDF passage answers one part")
            def extract(self, question, hits):
                return Extraction(claims=[Claim(chunk_id="chunk-one", source_id="infinigen-arxiv-v1", page=6,
                                                excerpt="KV pool in CPU memory",
                                                claim="InfiniGen keeps a KV pool in CPU memory")])
        def failed_web(*_):
            raise RuntimeError("network failure")
        result = research_questions([ResearchQuestion("combined", "paper and public status", "both")],
                                    lambda *_: [pdf], Reviewer(), target="InfiniGen", web_search=failed_web)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertTrue(any("웹 검색 실패" in note for note in result["notes"]))

    def test_both_can_keep_pdf_and_web_evidence(self):
        from langchain_core.documents import Document
        from kv_cache_eval.features.technical_research.infinigen.prompts import ResearchQuestion
        from kv_cache_eval.features.technical_research.infinigen.workflow import (
            Claim, Extraction, Review, research_questions,
        )

        pdf = self.hit()
        web = Document(id="web-two", page_content="The project page lists public code.", metadata={
            "chunk_id": "web-two", "source_id": "web-project", "source_title": "Project page",
            "source_url": "https://example.org/project", "source_kind": "web",
            "source_role": "web_external", "subject_technology": "InfiniGen", "page": None,
        })
        class Reviewer:
            def review(self, question, hits):
                return Review(sufficient=True, reason="two passages")
            def extract(self, question, hits):
                return Extraction(claims=[
                    Claim(chunk_id="chunk-one", source_id="infinigen-arxiv-v1", page=6,
                          excerpt="KV pool in CPU memory", claim="InfiniGen keeps a KV pool in CPU memory"),
                    Claim(chunk_id="web-two", source_id="web-project", page=None,
                          excerpt="lists public code", claim="The Project page lists public code."),
                ])
        result = research_questions([ResearchQuestion("combined", "paper and public", "both")],
                                    lambda *_: [pdf], Reviewer(), target="InfiniGen",
                                    web_search=lambda *_: ([web], []))
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual({item["source"]["page"] for item in result["evidence"]}, {6, None})


if __name__ == "__main__":
    unittest.main()
