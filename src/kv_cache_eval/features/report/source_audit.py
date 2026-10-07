"""Recheck primary PDF bodies before declaring experimental results unavailable."""

import re
from pathlib import Path

from kv_cache_eval.common.evidence import collect_evidence
from kv_cache_eval.common.tasks import EVALUATION_KEYS
from kv_cache_eval.features.technical_research.ingest import (
    DEFAULT_DOCUMENT_DIR,
    load_sources,
    validate_sources,
)


def collect_report_evidence(state, document_dir=DEFAULT_DOCUMENT_DIR):
    """Use the same hash-checked body pages for writer, Judge and PDF export.

    Only primary documents already represented in this run are audited. These
    pages are author-reported evidence, not independent validation. Stop at the
    reference heading; retrieved appendix evidence remains available separately.
    """
    evidence = collect_evidence(state)
    represented = {item["source"]["document"] for item in evidence.values()}
    for technology in state["selected_technologies"]:
        sources, budget = load_sources(technology)
        for source in sources:
            for eid, item in list(evidence.items()):
                if item["source"]["document"] in (
                    source.file,
                    source.citation_document,
                    source.document_id,
                ):
                    evidence[eid] = {
                        **item,
                        "source_title": source.title,
                        "source_version": source.version,
                    }
        primary = [
            source
            for source in sources
            if source.evidence_role == "primary"
            and represented.intersection(
                {source.file, source.citation_document, source.document_id}
            )
        ]
        if not primary:
            continue
        validate_sources(primary, Path(document_dir), budget)
        from pypdf import PdfReader

        for source in primary:
            for index, page in enumerate(
                PdfReader(Path(document_dir) / source.file).pages
            ):
                text = (page.extract_text() or "").strip()
                if re.search(r"^\s*References\s*$", text, re.MULTILINE | re.IGNORECASE):
                    break
                if not text:
                    continue
                eid = f"{technology.lower()}:source-audit:{source.document_id}:p{index + 1}"
                evidence[eid] = {
                    "id": eid,
                    "technology": technology,
                    "claim": f"{source.title}의 원문 본문 페이지. 수치와 해석은 발췌를 확인한다.",
                    "excerpt": text,
                    "source": {
                        "document": source.citation_document,
                        "url": source.source_url,
                        "page": index + 1,
                    },
                    "experiment": None,
                    "source_title": source.title,
                    "source_version": source.version,
                    "limitations": [
                        "저자 논문의 자체 보고이며 독립 검증이 아니다.",
                        "검색 결과의 근거 공백은 원문 전체의 부재를 의미하지 않는다.",
                    ],
                    "verification_status": "source_checked",
                }
    return evidence


def report_assessment_scope(state):
    """Keep scope, without turning retrieval-dependent scores into report facts."""
    return {
        key: [
            {
                field: row[field]
                for field in (
                    "technology",
                    "criterion",
                    "stakeholder_group",
                    "evidence_ids",
                )
                if field in row
            }
            for row in (state.get(key) or {}).get("evaluations", [])
        ]
        for key in EVALUATION_KEYS.values()
    }
