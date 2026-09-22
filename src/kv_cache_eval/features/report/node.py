"""입력: 종합과 실제 사용 근거. 출력: report 하나와 PDF 파일."""

import json
import os
import re
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

# 대괄호 안에 들어간 근거 ID 또는 잘못 생성된 근거 표시를 찾는다.
BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")


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


def _require_value(
    state: State,
    key: str,
) -> Any:
    """보고서 생성에 필요한 State 값이 존재하는지 확인한다."""
    value = state.get(key)

    if value is None:
        raise ValueError(
            f"{key}가 없습니다. 선행 노드가 완료된 후 "
            "report 노드를 실행해야 합니다."
        )

    return value


def _collect_evidence(
    state: State,
) -> dict[str, Evidence]:
    """기술 조사와 시장성 조사 결과를 근거 ID 기준으로 정리한다."""
    evidence_by_id: dict[str, Evidence] = {}

    for key in (
        "kivi_evidence",
        "infinigen_evidence",
        "market_evidence",
    ):
        research_result = state.get(key)

        if research_result is None:
            continue

        for evidence in research_result.get("evidence", []):
            evidence_id = evidence.get("id")

            if evidence_id:
                evidence_by_id[evidence_id] = evidence

    return evidence_by_id


def _reference_key(
    evidence: Evidence,
) -> tuple[str, ...]:
    """
    동일한 문서에서 생성된 여러 Evidence를 하나의 참고문헌으로 묶는다.

    URL이 있으면 URL을 우선 사용하고, URL이 없으면 기술명과 문서명을
    이용해 동일 문서 여부를 판단한다.
    """
    source = evidence["source"]
    url = str(source.get("url") or "").strip()
    document = str(
        source.get("document") or "문서명 미확인"
    ).strip()
    technology = str(evidence["technology"]).strip()

    if url:
        return (
            "url",
            url.rstrip("/"),
        )

    return (
        "document",
        technology.casefold(),
        document.casefold(),
    )


def _build_citation_index(
    cited_evidence_ids: list[str],
    evidence_by_id: dict[str, Evidence],
) -> tuple[dict[str, int], list[Evidence]]:
    """
    Evidence ID별 인용 번호와 중복 제거된 참고문헌 목록을 만든다.

    같은 URL 또는 같은 문서에서 생성된 Evidence는 동일한 참고문헌
    번호를 사용한다.
    """
    citation_numbers: dict[str, int] = {}
    reference_number_by_key: dict[tuple[str, ...], int] = {}
    reference_evidence: list[Evidence] = []

    for evidence_id in cited_evidence_ids:
        evidence = evidence_by_id.get(evidence_id)

        if evidence is None:
            continue

        reference_key = _reference_key(evidence)
        reference_number = reference_number_by_key.get(reference_key)

        if reference_number is None:
            reference_number = len(reference_evidence) + 1
            reference_number_by_key[reference_key] = reference_number
            reference_evidence.append(evidence)

        citation_numbers[evidence_id] = reference_number

    return citation_numbers, reference_evidence


def _format_reference(
    evidence: Evidence,
    reference_number: int,
) -> str:
    """Evidence를 사람이 읽을 수 있는 참고문헌 문자열로 변환한다."""
    technology = str(evidence["technology"]).strip()
    source = evidence["source"]

    document = str(
        source.get("document") or "문서명 미확인"
    ).strip()
    url = str(source.get("url") or "").strip()
    page = source.get("page")

    parts = [
        f"[{reference_number}]",
        f"{technology}.",
        f"{document}.",
    ]

    if not url and page is not None:
        parts.append(f"p. {page}.")

    if url:
        parts.append(url)

    return " ".join(parts)


def _build_reference_text(
    reference_evidence: list[Evidence],
) -> str:
    """중복이 제거된 Evidence를 REFERENCE 문자열로 만든다."""
    if not reference_evidence:
        return "보고서에 인용된 확인 가능 근거가 없습니다."

    references = [
        _format_reference(
            evidence=evidence,
            reference_number=index,
        )
        for index, evidence in enumerate(
            reference_evidence,
            start=1,
        )
    ]

    return "\n".join(references)


def _extract_section_evidence_ids(
    raw_report: ReportDraft,
    evidence_by_id: dict[str, Evidence],
) -> list[str]:
    """본문에 실제로 표시된 유효한 근거 ID를 등장 순서대로 찾는다."""
    extracted_ids: list[str] = []

    for section in raw_report.get("sections", []):
        if not isinstance(section, (list, tuple)):
            continue

        if len(section) != 2:
            continue

        content = str(section[1])

        for bracket_content in BRACKET_PATTERN.findall(content):
            evidence_id = bracket_content.strip()

            if evidence_id not in evidence_by_id:
                continue

            if evidence_id in extracted_ids:
                continue

            extracted_ids.append(evidence_id)

    return extracted_ids


def _looks_like_invalid_citation(
    bracket_content: str,
) -> bool:
    """
    LLM이 생성한 근거 placeholder인지 판단한다.

    일반적인 문서 제목 표현인 [ICML 2024] 등은 유지하고,
    근거 ID처럼 보이거나 미확인 placeholder인 경우만 제거한다.
    """
    normalized = bracket_content.strip().casefold()

    if ":" in normalized:
        return True

    if normalized.startswith("market-"):
        return True

    placeholder_keywords = (
        "evidence",
        "미확인",
        "확인 필요",
        "notes",
        "stakeholder",
        "project",
        "source",
        "citation",
        "근거 id",
        "trl notes",
    )

    return any(
        keyword in normalized
        for keyword in placeholder_keywords
    )


def _merge_numbered_citations(
    text: str,
) -> str:
    """연속된 번호 인용을 [1, 2] 형식으로 합친다."""
    consecutive_pattern = re.compile(
        r"(?:\[\d+\]\s*){2,}"
    )

    def replace_group(
        match: re.Match[str],
    ) -> str:
        numbers = re.findall(
            r"\d+",
            match.group(0),
        )

        unique_numbers = list(dict.fromkeys(numbers))

        return f"[{', '.join(unique_numbers)}]"

    return consecutive_pattern.sub(
        replace_group,
        text,
    ).strip()


def _replace_internal_citations(
    text: str,
    citation_numbers: dict[str, int],
) -> str:
    """
    본문의 내부 근거 ID를 사람이 읽는 번호 인용으로 변환한다.

    검증되지 않은 내부 ID나 placeholder는 [근거 미확인]으로
    표시한다.
    """

    def replace_match(
        match: re.Match[str],
    ) -> str:
        bracket_content = match.group(1).strip()

        # 이미 [1], [1, 2] 형태로 변환된 인용은 유지한다.
        if re.fullmatch(
            r"\d+(?:\s*,\s*\d+)*",
            bracket_content,
        ):
            return match.group(0)

        reference_number = citation_numbers.get(
            bracket_content
        )

        if reference_number is not None:
            return f"[{reference_number}]"

        if _looks_like_invalid_citation(bracket_content):
            return "[근거 미확인]"

        # [ICML 2024]와 같은 일반적인 대괄호 표현은 유지한다.
        return match.group(0)

    converted = BRACKET_PATTERN.sub(
        replace_match,
        text,
    )

    return _merge_numbered_citations(converted)


def _normalize_sections(
    raw_report: ReportDraft,
    reference_text: str,
    citation_numbers: dict[str, int],
) -> list[tuple[str, str]]:
    """LLM 결과를 지정된 보고서 목차 순서로 정규화한다."""
    generated_sections: dict[str, str] = {}

    for section in raw_report.get("sections", []):
        if not isinstance(section, (list, tuple)):
            continue

        if len(section) != 2:
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
            content = generated_sections.get(
                title,
                "",
            ).strip()

            if not content:
                content = (
                    "해당 항목의 분석 결과가 생성되지 않았습니다."
                )

            content = _replace_internal_citations(
                text=content,
                citation_numbers=citation_numbers,
            )

        normalized_sections.append(
            (
                title,
                content,
            )
        )

    return normalized_sections


def _normalize_report(
    raw_report: ReportDraft,
    synthesis: Synthesis,
    evidence_by_id: dict[str, Evidence],
) -> ReportDraft:
    """보고서의 근거 ID와 섹션을 State 스키마에 맞게 정리한다."""
    requested_ids = _extract_section_evidence_ids(
        raw_report=raw_report,
        evidence_by_id=evidence_by_id,
    )

    requested_ids.extend(
        raw_report.get(
            "cited_evidence_ids",
            [],
        )
    )

    requested_ids.extend(
        synthesis.get(
            "cited_evidence_ids",
            [],
        )
    )

    cited_ids = [
        evidence_id
        for evidence_id in dict.fromkeys(requested_ids)
        if evidence_id in evidence_by_id
    ]

    citation_numbers, reference_evidence = (
        _build_citation_index(
            cited_evidence_ids=cited_ids,
            evidence_by_id=evidence_by_id,
        )
    )

    reference_text = _build_reference_text(
        reference_evidence=reference_evidence,
    )

    sections = _normalize_sections(
        raw_report=raw_report,
        reference_text=reference_text,
        citation_numbers=citation_numbers,
    )

    return {
        "sections": sections,
        "cited_evidence_ids": cited_ids,
    }


def _paragraph_text(
    text: str,
) -> str:
    """ReportLab Paragraph에서 안전하게 표시할 문자열로 변환한다."""
    escaped_text = escape(text)

    return escaped_text.replace(
        "\n",
        "<br/>",
    )


def _resolve_pdf_path() -> Path:
    """환경변수가 있으면 해당 경로를, 없으면 기본 경로를 사용한다."""
    configured_path = os.getenv(
        "REPORT_PDF_PATH",
        "",
    ).strip()

    if configured_path:
        return Path(configured_path).expanduser()

    return DEFAULT_PDF_PATH


def _resolve_font_path() -> Path:
    """프로젝트에 포함된 한글 폰트 파일의 경로를 반환한다."""
    configured_path = os.getenv(
        "REPORT_FONT_PATH",
        "",
    ).strip()

    if configured_path:
        font_path = Path(configured_path).expanduser()
    else:
        font_path = (
            Path(__file__).resolve().parents[4]
            / "assets"
            / "fonts"
            / "NanumGothic.ttf"
        )

    if not font_path.exists():
        raise FileNotFoundError(
            "한글 PDF 생성을 위한 폰트 파일을 찾을 수 없습니다: "
            f"{font_path}"
        )

    return font_path


def _write_pdf(
    report: ReportDraft,
    output_path: Path,
) -> None:
    """구조화된 보고서를 한글 PDF 파일로 저장한다."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    font_path = _resolve_font_path()
    font_name = "NanumGothic"

    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(
            TTFont(
                font_name,
                str(font_path),
            )
        )

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
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        name="KoreanSubtitle",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=11,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=10,
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
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        name="KoreanBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=16,
        alignment=TA_LEFT,
        wordWrap="CJK",
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )

    reference_style = ParagraphStyle(
        name="KoreanReference",
        parent=body_style,
        fontSize=8.5,
        leading=14,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#374151"),
    )

    story = [
        Paragraph(
            "KV Cache 최적화 기술 다관점 평가 보고서",
            title_style,
        ),
        Paragraph(
            "KIVI와 InfiniGen 비교",
            subtitle_style,
        ),
        Spacer(
            1,
            5 * mm,
        ),
    ]

    for section_title, section_content in report["sections"]:
        if section_title == "SUMMARY":
            heading = summary_title_style
            content_style = body_style
        elif section_title == "REFERENCE":
            # 강제 PageBreak를 사용하지 않고 한계점 뒤에 이어서 출력한다.
            heading = heading_style
            content_style = reference_style
        else:
            heading = heading_style
            content_style = body_style

        story.append(
            Paragraph(
                escape(section_title),
                heading,
            )
        )

        story.append(
            Paragraph(
                _paragraph_text(section_content),
                content_style,
            )
        )

        if section_title == "SUMMARY":
            story.append(
                Spacer(
                    1,
                    4 * mm,
                )
            )

    document.build(story)


def write_report(
    state: State,
) -> StateUpdate:
    """종합 결과와 근거를 이용해 최종 보고서와 PDF를 생성한다."""
    synthesis = _require_value(
        state,
        "synthesis",
    )
    maturity_eval = _require_value(
        state,
        "maturity_eval",
    )
    market_eval = _require_value(
        state,
        "market_eval",
    )
    stakeholder_eval = _require_value(
        state,
        "stakeholder_eval",
    )
    domain_eval = _require_value(
        state,
        "domain_eval",
    )

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
7. [project evidence], [TRL notes], [stakeholder 미확인]과 같은
   임시 근거 표시를 절대 생성하지 마십시오.
8. 공개 자료 기반 TRL은 추정이라는 점을 명시하십시오.
9. 논문 환경과 실제 운영 환경의 차이를 한계에 포함하십시오.
10. REFERENCE 내용은 코드에서 실제 인용 근거로 다시 작성하므로,
    임의의 자료를 추가하지 마십시오.
11. 모든 본문은 한국어로 작성하십시오.

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

    structured_llm = _get_llm().with_structured_output(
        ReportDraft,
        method="function_calling",
    )

    raw_report = structured_llm.invoke(prompt)

    if not isinstance(raw_report, dict):
        raise TypeError(
            "LLM이 ReportDraft 형식의 결과를 반환하지 않았습니다."
        )

    report = _normalize_report(
        raw_report=raw_report,
        synthesis=synthesis,
        evidence_by_id=evidence_by_id,
    )

    pdf_path = _resolve_pdf_path()

    _write_pdf(
        report=report,
        output_path=pdf_path,
    )

    return {
        "report": report,
    }