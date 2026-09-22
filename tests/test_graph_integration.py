"""외부 응답만 대체하고 실제 그래프와 평가·보고서 흐름을 검증한다."""

import hashlib
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.domain.prompt import OUTPUT_SCHEMA, VERIFY_SCHEMA
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC
from kv_cache_eval.features.market.analysis import CitedAssessment, MarketAnalysis
from kv_cache_eval.features.market.node import build_market_node
from kv_cache_eval.features.report.node import (
    REPORT_SECTION_TITLES, _ReportResponse, _normalize_report,
)
from kv_cache_eval.features.stakeholders.criteria import STAKEHOLDER_CRITERIA
from kv_cache_eval.features.stakeholders.node import _StakeholderAssessmentBatch, evaluate as evaluate_stakeholders
from kv_cache_eval.features.synthesis.node import _SynthesisResponse, _validate_synthesis
from kv_cache_eval.features.technical_research.workflow import Claim, Extraction, Review
from kv_cache_eval.graph import build_graph


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PDF = ROOT / "output/pdf/graph_integration_offline_example.pdf"
_SOURCES = {
    "KIVI": ("kivi_original", "kivi_original.pdf", "KIVI fixture describes a cache method."),
    "InfiniGen": ("infinigen-arxiv-v1", "infinigen-arxiv-v1", "InfiniGen fixture describes a cache method."),
}


def _paper(technology, *, principle=False):
    source_id, document, content = _SOURCES[technology]
    if principle:
        content += " Principle fixture."
    return Document(page_content=content, metadata={
        "chunk_id": f"{technology.lower()}-paper-{'principle' if principle else 'general'}", "source_id": source_id,
        "source_document": document, "source_title": document,
        "source_url": "https://example.com/fixture-paper", "page": 1,
        "source_role": "primary", "subject_technology": technology,
        "source_kind": "pdf",
    })


def _web(technology):
    url = f"https://example.com/{technology.lower()}-fixture"
    return Document(page_content=f"{technology} fixture public page.", metadata={
        "chunk_id": f"{technology.lower()}-web", "source_id": url,
        "source_title": f"{technology} fixture page", "source_url": url,
        "page": None, "source_role": "web_external",
        "subject_technology": technology, "source_kind": "web",
    })


class _Retriever:
    def __init__(self, technology):
        self.technology = technology

    def search(self, query, count):
        return [_paper(self.technology, principle="핵심 설계 원리" in query)]


def _review(self, question, hits):
    return Review(sufficient=True, reason="fixture excerpt found")


def _extract(self, question, hits):
    hit = hits[0]
    meta = hit.metadata
    return Extraction(claims=[Claim(
        chunk_id=meta["chunk_id"], source_id=meta["source_id"],
        page=meta["page"], excerpt=hit.page_content,
        claim=hit.page_content,
    )])


def _market_search(query):
    digest = hashlib.sha256(query.encode()).hexdigest()[:12]
    return [{"title": "오프라인 시장 예시", "url": f"https://example.com/market/{digest}",
             "content": "오프라인 검증용 예시", "raw_content": "오프라인 검증용 예시"}]


def _market_analyse(technology, hits):
    urls = [hit.url for hit in hits]
    return MarketAnalysis(
        cagr_percent=20,
        growth=CitedAssessment(judgment="예시 시장 자료", source_urls=[urls[0]]),
        adoption_level="public_prototype_only",
        adoption=CitedAssessment(judgment="예시 공개 구현", source_urls=[urls[-2]]),
        supports=[],
        ecosystem=CitedAssessment(judgment="예시 생태계", source_urls=[urls[-1]]),
    )


def _stakeholder_call(*, technology, domain, evidence):
    return _StakeholderAssessmentBatch.model_validate({
        "assessments": [{
            "stakeholder_group": item.group, "judgment": "오프라인 검증용 추론",
            "rationale": "fixture 근거를 이용한 추론", "score": None,
            "evidence_ids": [evidence[0]["id"]], "basis_status": "inferred",
        } for item in STAKEHOLDER_CRITERIA],
    })


def _domain_invoke(messages, schema, *, leave_gaps=False):
    import json
    payload = json.loads(messages[1][1])
    if schema is OUTPUT_SCHEMA:
        if leave_gaps:
            return {"evaluations": [], "notes": []}
        first = {tech: next(item for item in payload["evidence"] if item["technology"] == tech)
                 for tech in payload["technologies"]}
        return {"evaluations": [{
            "technology": tech, "criterion": criterion,
            "judgment": "오프라인 검증용 판단", "score": None,
            "rationale": "fixture 인용문을 확인했다.",
            "supports": [{"evidence_id": first[tech]["id"], "quote": first[tech]["claim"]}],
            "measurement": None, "uncertainty": "실제 운영 검증 없음",
        } for tech in payload["technologies"] for criterion in DOMAIN_RUBRIC], "notes": []}
    if schema is VERIFY_SCHEMA:
        return {"reviews": [{
            "technology": draft["technology"], "criterion": draft["criterion"],
            "supported": True, "measurement_supported": False,
            "rubric_supported": True, "reason": "fixture 일치",
        } for draft in payload["drafts"]]}
    raise AssertionError("예상하지 않은 도메인 응답 스키마")


class _FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt):
        import json
        if self.schema is _SynthesisResponse:
            payload = json.loads(prompt.split("입력 데이터:\n", 1)[1].split("\n\n다음 항목", 1)[0])
            available = payload["valid_evidence_ids"]
        else:
            payload = json.loads(prompt.split("입력 데이터:\n", 1)[1].split("\n\n다음 구조", 1)[0])
            available = list(payload["evidence"])
        citation = next((eid for eid in available if ":principle:" in eid), available[0])
        if self.schema is _SynthesisResponse:
            return self.schema.model_validate({
                "perspective_differences": [f"오프라인 검증용 차이 [{citation}]"],
                "tradeoffs": ["오프라인 검증용 상충 관계"],
                "application_conditions": ["실제 환경은 미확인"],
                "cited_evidence_ids": [citation],
            })
        if self.schema is _ReportResponse:
            return self.schema.model_validate({
                "sections": [{"title": title, "content": (
                    f"오프라인 검증용 예시입니다. 실제 성능이나 채택을 뜻하지 않습니다. [{citation}]"
                    if title == "SUMMARY" else "오프라인 검증용 예시입니다. 실제 조사 결과가 아닙니다."
                )} for title in REPORT_SECTION_TITLES],
                "cited_evidence_ids": [citation],
            })
        raise AssertionError("예상하지 않은 LLM 응답 스키마")


class _FakeLLM:
    def with_structured_output(self, schema, **kwargs):
        self._schema_options = kwargs
        return _FakeStructured(schema)


class GraphIntegrationTest(unittest.TestCase):
    def test_report_response_schema_uses_section_objects(self):
        schema = _ReportResponse.model_json_schema()
        section = schema["$defs"]["_ReportSection"]
        self.assertEqual(section["type"], "object")
        self.assertEqual(set(section["properties"]), {"title", "content"})

    def _run(self, pdf_path, *, max_research_rounds, leave_gaps=False):
        market_node = build_market_node(_market_search, _market_analyse)
        graph = build_graph({
            "market": market_node,
            "stakeholders": lambda state: evaluate_stakeholders(state, llm_call=_stakeholder_call),
        }, user_question="오프라인 검증용 질문")
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")), \
             patch("socket.create_connection", side_effect=AssertionError("network called")), \
             patch("kv_cache_eval.features.technical_research.node.make_runtime_llm", return_value=object()), \
             patch("kv_cache_eval.features.technical_research.retriever.ensure_index",
                   side_effect=lambda technology, **kwargs: (_Retriever(technology), {})), \
             patch("kv_cache_eval.features.technical_research.web.search_web",
                   side_effect=lambda query, technology, count: ([_web(technology)], [])), \
             patch("kv_cache_eval.features.technical_research.workflow.ModelReviewer.review", _review), \
             patch("kv_cache_eval.features.technical_research.workflow.ModelReviewer.extract", _extract), \
             patch("kv_cache_eval.features.domain.node.invoke_structured",
                   side_effect=lambda messages, schema: _domain_invoke(messages, schema, leave_gaps=leave_gaps)), \
             patch("kv_cache_eval.features.synthesis.node._get_llm", return_value=_FakeLLM()), \
             patch("kv_cache_eval.features.report.node._get_llm", return_value=_FakeLLM()), \
             patch.dict(os.environ, {"REPORT_PDF_PATH": str(pdf_path)}):
            return graph.invoke(new_state(max_research_rounds=max_research_rounds))

    def test_full_graph_finishes_and_writes_example_pdf(self):
        final = self._run(EXAMPLE_PDF, max_research_rounds=0)
        self.assertEqual(final["research_round"], 0)
        self.assertEqual(final["evidence_gaps"], [])
        self.assertEqual([title for title, _ in final["report"]["sections"]], list(REPORT_SECTION_TITLES))
        citation = final["report"]["cited_evidence_ids"][0]
        self.assertIn(":principle:", citation)
        self.assertIn(citation, final["report"]["sections"][-1][1])
        self.assertTrue(EXAMPLE_PDF.is_file())

    def test_retry_limit_keeps_every_unresolved_gap_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "offline_retry_example.pdf"
            final = self._run(pdf_path, max_research_rounds=1, leave_gaps=True)
            self.assertTrue(pdf_path.is_file())
        self.assertEqual(final["research_round"], 1)
        self.assertTrue(final["evidence_gaps"])
        self.assertEqual(final["synthesis"]["unresolved_gaps"], final["evidence_gaps"])
        limitations = dict(final["report"]["sections"])["6. 분석 한계 및 편향 방지"]
        self.assertIn("미해결 근거 공백", limitations)
        for gap in final["evidence_gaps"]:
            self.assertIn(gap["reason"], limitations)

    def test_unknown_citation_is_rejected_before_reference_or_pdf(self):
        evidence = {"known": {"id": "known"}}
        raw = {"sections": [("SUMMARY", "검증되지 않은 주장 [invented]"),
                            ("REFERENCE", "")], "cited_evidence_ids": []}
        synthesis = {"cited_evidence_ids": []}
        with self.assertRaisesRegex(ValueError, "invented"):
            _normalize_report(raw, synthesis, evidence, [])
        with self.assertRaisesRegex(ValueError, "invented"):
            _validate_synthesis({"perspective_differences": ["주장 [invented]"]}, {"known"}, [])

    def test_reference_uses_only_citations_in_final_sections(self):
        evidence = {name: {
            "id": name, "technology": "KIVI",
            "source": {"document": "fixture.pdf", "url": None, "page": 1},
            "limitations": [],
        } for name in ("kept", "dropped")}
        raw = {"sections": [
            ("SUMMARY", "이전 초안 [dropped]"),
            ("알 수 없는 제목", "버려지는 문단 [dropped]"),
            ("SUMMARY", "최종 요약 [kept]"),
        ], "cited_evidence_ids": ["dropped", "kept"]}
        report = _normalize_report(raw, {"cited_evidence_ids": []}, evidence, [])
        self.assertEqual(report["cited_evidence_ids"], ["kept"])
        self.assertIn("[kept]", report["sections"][-1][1])
        self.assertNotIn("[dropped]", report["sections"][-1][1])


if __name__ == "__main__":
    unittest.main()
