"""입력: 종합과 실제 사용 근거. 출력: report 하나와 PDF 파일."""

import json
import os
from html import escape
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.schemas import Evidence, ReportDraft, Synthesis
from kv_cache_eval.common.state import State, StateUpdate


REPORT_SECTION_TITLES = (
    "SUMMARY",
    "1. 분석 배경",
    "2. 기술 선정 및 개요",
    "3. 평가 기준 및 방법",
    "4. 관점별 평가 결과",
    "5. 종합 평가 및 시사점",
    "6. 한계점",
    "REFERENCE",
)

DEFAULT_PDF_PATH = Path("outputs/kv_cache_evaluation_report.pdf")


def _get_llm() -> ChatOpenAI:
    """환경변수에 지정된 OpenAI 생성 모델을 만든다."""
    load_environment()

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if provider != "openai":
        raise ValueError(
            "report 노드는 LLM_PROVIDER=openai 설정이 필요합니다."
        )

    if not model_name:
        raise ValueError("LLM_MODEL 환경변수가 설정되지 않았습니다.")

    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
    )


def _require_value(state: State, key: str) -> Any:
    """보고서 생성에 필요한 State 값이 존재하는지 확인한다."""
    value = state.get(key)

    if value is None:
        raise ValueError(
            f"{key}가 없습니다. 선행 노드가 완료된 후 "
            "report 노드를 실행해야 합니다."
        )

    return value


def _collect_evidence(state: State) -> dict[str, Evidence]:
    """두 기술의 조사 결과를 근거 ID 기준으로 정리한다."""
    evidence_by_id: dict[str, Evidence] = {}

    for key in ("kivi_evidence", "infinigen_evidence"):
        research_result = state.get(key)

        if research_result is None:
            continue

        for evidence in research_result.get("evidence", []):
            evidence_id = evidence.get("id")

            if evidence_id:
                evidence_by_id[evidence_id] = evidence

    return evidence_by_id


def _format_reference(evidence: Evidence) -> str:
    """Evidence를 REFERENCE에 사용할 문자열로 변환한다."""
    evidence_id = evidence["id"]
    technology = evidence["technology"]
    source = evidence["source"]

    document = source.get("document") or "문서명 미확인"
    url = source.get("url")
    page = source.get("page")

    parts = [f"[{evidence_id}] {technology}. {document}."]

    if page is not None:
        parts.append(f"p. {page}.")

    if url:
        parts.append(url)

    limitations = evidence.get("limitations", [])

    if limitations:
        parts.append(
            "근거 한계: " + "; ".join(str(item) for item in limitations)
        )

    return " ".join(parts)


def _build_reference_text(
    cited_evidence_ids: list[str],
    evidence_by_id: dict[str, Evidence],
) -> str:
    """실제로 인용된 근거만 REFERENCE 문자열로 만든다."""
    references: list[str] = []

    for evidence_id in cited_evidence_ids:
        evidence = evidence_by_id.get(evidence_id)

        if evidence is None:
            continue

        references.append(_format_reference(evidence))

    if not references:
        return "보고서에 인용된 확인 가능 근거가 없습니다."

    return "\n".join(
        f"{index}. {reference}"
        for index, reference in enumerate(references, start=1)
    )


def _normalize_sections(
    raw_report: ReportDraft,
    reference_text: str,
) -> list[tuple[str, str]]:
    """LLM 결과를 지정된 보고서 목차 순서로 정규화한다."""
    generated_sections: dict[str, str] = {}

    for section in raw_report.get("sections", []):
        if not isinstance(section, (list, tuple)) or len(section) != 2:
            continue

        title = str(section[0]).strip()
        content = str(section[1]).strip()

        if title:
            generated_sections[title] = content

    normalized_sections: list[tuple[str, str]] = []

    for title in REPORT_SECTION_TITLES:
        if title == "REFERENCE":
            content = reference_text
        else:
            content = generated_sections.get(title, "").strip()

            if not content:
                content = "해당 항목의 분석 결과가 생성되지 않았습니다."

        normalized_sections.append((title, content))

    return normalized_sections


def _normalize_report(
    raw_report: ReportDraft,
    synthesis: Synthesis,
    evidence_by_id: dict[str, Evidence],
) -> ReportDraft:
    """보고서의 근거 ID와 섹션을 State 스키마에 맞게 정리한다."""
    requested_ids = list(raw_report.get("cited_evidence_ids", []))
    requested_ids.extend(synthesis.get("cited_evidence_ids", []))

    cited_ids = [
        evidence_id
        for evidence_id in dict.fromkeys(requested_ids)
        if evidence_id in evidence_by_id
    ]

    reference_text = _build_reference_text(
        cited_evidence_ids=cited_ids,
        evidence_by_id=evidence_by_id,
    )

    sections = _normalize_sections(
        raw_report=raw_report,
        reference_text=reference_text,
    )

    return {
        "sections": sections,
        "cited_evidence_ids": cited_ids,
    }


def _paragraph_text(text: str) -> str:
    """ReportLab Paragraph에서 안전하게 표시할 문자열로 변환한다."""
    escaped_text = escape(text)
    return escaped_text.replace("\n", "<br/>")


def _resolve_pdf_path() -> Path:
    """환경변수가 있으면 해당 경로를, 없으면 기본 경로를 사용한다."""
    configured_path = os.getenv("REPORT_PDF_PATH", "").strip()

    if configured_path:
        return Path(configured_path).expanduser()

    return DEFAULT_PDF_PATH


def _write_pdf(report: ReportDraft, output_path: Path) -> None:
    """구조화된 보고서를 한글 PDF 파일로 저장한다."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF 생성을 위해 reportlab이 필요합니다. "
            "`uv sync --locked --extra llm-openai --extra report`를 "
            "실행하십시오."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ReportLab에서 제공하는 한국어 CID 폰트를 사용한다.
    # 운영체제별 로컬 폰트 경로에 의존하지 않으므로 팀 환경에서 사용하기 쉽다.
    font_name = "HYSMyeongJo-Medium"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="KIVI와 InfiniGen 다관점 평가 보고서",
        author="SKALA-RAG-KV-cache",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="KoreanTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=12,
    )

    summary_title_style = ParagraphStyle(
        name="KoreanSummaryTitle",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=15,
        leading=22,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#111827"),
        spaceBefore=8,
        spaceAfter=8,
    )

    heading_style = ParagraphStyle(
        name="KoreanHeading",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=14,
        leading=21,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        name="KoreanBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=16,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        spaceAfter=8,
    )

    reference_style = ParagraphStyle(
        name="KoreanReference",
        parent=body_style,
        fontSize=8.5,
        leading=14,
        alignment=TA_LEFT,
    )

    story = [
        Paragraph(
            "KV Cache 최적화 기술 다관점 평가 보고서",
            title_style,
        ),
        Paragraph(
            "KIVI와 InfiniGen 비교",
            body_style,
        ),
        Spacer(1, 6 * mm),
    ]

    for index, (section_title, section_content) in enumerate(
        report["sections"]
    ):
        if section_title == "SUMMARY":
            heading = summary_title_style
            content_style = body_style
        elif section_title == "REFERENCE":
            # REFERENCE는 보고서의 마지막 장에서 시작한다.
            if index > 0:
                story.append(PageBreak())

            heading = heading_style
            content_style = reference_style
        else:
            heading = heading_style
            content_style = body_style

        story.append(Paragraph(escape(section_title), heading))
        story.append(
            Paragraph(
                _paragraph_text(section_content),
                content_style,
            )
        )

        if section_title == "SUMMARY":
            # SUMMARY는 프롬프트에서 500~700자로 제한하고,
            # 이후 본문과 시각적으로 구분한다.
            story.append(Spacer(1, 5 * mm))

    document.build(story)


def write_report(state: State) -> StateUpdate:
    """종합 결과와 근거를 이용해 최종 보고서와 PDF를 생성한다."""
    synthesis = _require_value(state, "synthesis")
    maturity_eval = _require_value(state, "maturity_eval")
    market_eval = _require_value(state, "market_eval")
    stakeholder_eval = _require_value(state, "stakeholder_eval")
    domain_eval = _require_value(state, "domain_eval")

    evidence_by_id = _collect_evidence(state)

    report_input = {
        "selected_technologies": state["selected_technologies"],
        "domain_and_criteria": state["domain_and_criteria"],
        "synthesis": synthesis,
        "evaluations": {
            "maturity": maturity_eval,
            "market": market_eval,
            "stakeholder": stakeholder_eval,
            "domain": domain_eval,
        },
        "evidence": evidence_by_id,
    }

    prompt = f"""
당신은 KV cache 최적화 기술 다관점 평가 보고서 작성자입니다.

비교 대상:
- KIVI: KV cache 2비트 비대칭 양자화 기반 소프트웨어 접근
- InfiniGen: 필요한 KV 항목을 선택적으로 프리패치하는
  호스트 메모리 오프로딩 접근

평가 도메인:
- GPU 기반 클라우드 LLM 서비스

다음 원칙을 반드시 지키십시오.
1. 특정 기술을 추천하거나 승자를 결정하지 마십시오.
2. 관점에 따라 평가가 달라지는 지점을 중립적으로 기술하십시오.
3. 확인된 사실, 공개 정보 기반 추정, 분석상 추론,
   미확인 정보를 구분하십시오.
4. 입력에 존재하지 않는 도입 사례나 성능 수치를 만들지 마십시오.
5. 근거를 사용할 때는 문장에 [근거 ID] 형식으로 표시하십시오.
6. 제공된 evidence에 없는 근거 ID를 만들지 마십시오.
7. 공개 자료 기반 TRL은 추정이라는 점을 명시하십시오.
8. 논문 환경과 실제 운영 환경의 차이를 한계에 포함하십시오.
9. REFERENCE 내용은 코드에서 실제 인용 근거로 다시 작성하므로,
   임의의 자료를 추가하지 마십시오.
10. 모든 본문은 한국어로 작성하십시오.

보고서 섹션은 반드시 다음 순서와 정확한 제목을 사용하십시오.
1. SUMMARY
2. 1. 분석 배경
3. 2. 기술 선정 및 개요
4. 3. 평가 기준 및 방법
5. 4. 관점별 평가 결과
6. 5. 종합 평가 및 시사점
7. 6. 한계점
8. REFERENCE

작성 기준:
- SUMMARY는 개요 목록이 아니라 전체 평가 결과의 핵심 요약입니다.
- SUMMARY는 PDF 반 페이지를 넘지 않도록 약 500~700자로 작성하십시오.
- 기술 선정 및 개요에는 KIVI와 InfiniGen의 선정 이유,
  핵심 원리와 한계를 포함하십시오.
- 관점별 평가 결과에는 기술 성숙도, 시장성, 이해관계자,
  도메인 적용성을 모두 포함하십시오.
- 종합 평가 및 시사점에는 공통점, 차이점, 상충 관계,
  기술별 적용 조건을 포함하십시오.
- 한계점에는 공개 정보 기반 분석, 논문과 운영 환경의 차이,
  확증편향을 줄이기 위한 조치를 포함하십시오.
- REFERENCE 섹션은 빈 문자열로 두어도 됩니다.
  실제 사용된 근거만 코드에서 입력합니다.

입력 데이터:
{json.dumps(report_input, ensure_ascii=False, indent=2)}

ReportDraft 구조에 맞춰 다음 값을 생성하십시오.
- sections: (섹션 제목, 본문) 쌍의 목록
- cited_evidence_ids: 보고서 작성에 실제 사용한 근거 ID 목록
"""

    structured_llm = _get_llm().with_structured_output(ReportDraft)
    raw_report = structured_llm.invoke(prompt)

    if not isinstance(raw_report, dict):
        raise TypeError("LLM이 ReportDraft 형식의 결과를 반환하지 않았습니다.")

    report = _normalize_report(
        raw_report=raw_report,
        synthesis=synthesis,
        evidence_by_id=evidence_by_id,
    )

    pdf_path = _resolve_pdf_path()
    _write_pdf(report=report, output_path=pdf_path)

    return {"report": report}