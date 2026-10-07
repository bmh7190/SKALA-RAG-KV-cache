"""Focused safeguards for the four mandatory findings in the report review."""

import re
from pathlib import Path

from kv_cache_eval.common.evidence import collect_evidence
from kv_cache_eval.features.technical_research.ingest import (
    DEFAULT_DOCUMENT_DIR,
    load_sources,
    validate_sources,
)

# Physical pages: abstracts, benchmark names, throughput/latency conditions.
RECHECK_PAGES = {"KIVI": (1, 2, 8, 9), "InfiniGen": (1, 9, 11, 12)}

FACTUAL_RULES = """다음 필수 사실 검증 규칙을 적용한다:
1. InfiniGen의 데이터셋 이름은 PTB(Penn Treebank)다. PTT로 쓰지 않는다.
2. 처리량(tokens/s) 증가나 최대 배치 크기 증가는 개별 요청의 지연 시간 감소를
입증하지 않는다. KIVI의 2.35~3.47배 처리량 결과로 지연 개선을 직접·간접 확인했다고
쓰거나 지연 감소 가능성의 실험 근거로 삼지 않는다. 지연은 별도 측정이 필요하다.
3. InfiniGen 초록의 최대 3.00배 요약과 Figure 14의 OPT-13B, 입력 1920토큰,
출력 128토큰, 배치 20 실험을 구분한다. 후자의 원문 결과는 비교 baseline들에 대한
1.63~32.93배 범위다. 초록의 3.00배를 이 실험의 결과로 단정하지 않는다.
수치를 쓸 때 원문에서 확인한 비교 대상과 조건을 함께 적고, 확인할 수 없으면 생략한다.
4. 두 기술의 병목·설계 방식이 다르다는 이유로 서로 다른 시장을 겨냥한다고 단정하지
않는다. 모두 LLM 추론에 적용될 수 있으며 시장 분리를 입증한 근거는 별도로 필요하다.
이전 평가·종합 결과와 이 규칙이 충돌하면 실제 인용 원문을 다시 확인한다.
품질 평가자는 이 네 항목을 groundedness에서 검사하고, 틀린 주장이나 근거 없는
인과·시장 단정을 발견하면 fail로 반환한다. 인용 ID의 존재만으로 통과시키지 않는다.
검사 규칙은 보고서 본문이 아니다. 보고서에 실제로 없는 주장을 있다고 간주하지 않는다.
이 수치나 데이터셋을 반드시 본문에 추가할 필요는 없으며, 언급하지 않은 것 자체를
사실 오류로 판정하지 않는다. fail 이유에는 실제 본문의 문제 문장과 대응 원문을 적는다."""


def collect_report_evidence(state, document_dir=DEFAULT_DOCUMENT_DIR):
    """Add only the primary pages needed to recheck the mandatory findings.

    Documents must already be represented by verified research evidence; their
    hash and page count are checked. Writer, Judge and exporter share these IDs.
    """
    evidence = collect_evidence(state)
    represented = {item["source"]["document"] for item in evidence.values()}
    for technology, pages in RECHECK_PAGES.items():
        sources, budget = load_sources(technology)
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
            reader = PdfReader(Path(document_dir) / source.file)
            for page in pages:
                text = (reader.pages[page - 1].extract_text() or "").strip()
                if not text:
                    raise ValueError(
                        f"필수 검증 원문이 비어 있습니다: {source.document_id}:p{page}"
                    )
                eid = f"{technology.lower()}:fact-check:{source.document_id}:p{page}"
                evidence[eid] = {
                    "id": eid,
                    "technology": technology,
                    "claim": "필수 사실 검증용 원문 페이지. 주장과 실험 조건은 발췌에서 확인한다.",
                    "excerpt": text,
                    "source": {
                        "document": source.citation_document,
                        "url": source.source_url,
                        "page": page,
                    },
                    "experiment": None,
                    "limitations": ["논문 저자의 자체 보고이며 독립 검증이 아니다."],
                    "verification_status": "source_checked",
                }
    return evidence


def factual_issues(report):
    """Reject the unambiguous dataset typo; contextual claims go to the Judge."""
    return [
        (title, "데이터셋 이름 PTT는 오기입니다", "PTB(Penn Treebank)로 수정한다")
        for title, text in report["sections"]
        if title != "REFERENCE" and re.search(r"(?<![A-Za-z])PTT(?![A-Za-z])", text)
    ]
