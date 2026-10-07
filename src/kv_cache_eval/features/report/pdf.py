"""Render and atomically publish a quality-approved report."""
import os
from pathlib import Path
from html import escape
from kv_cache_eval.common.schemas import ReportDraft

DEFAULT_PDF_PATH = Path("output/pdf/kv_cache_evaluation_report.pdf")

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


