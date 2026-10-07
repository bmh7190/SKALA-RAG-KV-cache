"""Render and atomically publish a quality-approved report."""

import os
from datetime import date
from html import escape
from pathlib import Path

from kv_cache_eval.common.schemas import ReportDraft

DEFAULT_PDF_PATH = Path("output/pdf/kv_cache_evaluation_report.pdf")


def _cover_metadata():
    """Use explicit authors only; never infer team membership from repository history."""
    values = {
        "authors": os.getenv("REPORT_AUTHORS", "").strip(),
        "affiliation": os.getenv("REPORT_AFFILIATION", "SKALA").strip(),
        "date": os.getenv("REPORT_DATE", "").strip() or date.today().isoformat(),
    }
    for key, limit in (("authors", 500), ("affiliation", 150), ("date", 80)):
        if len(values[key]) > limit:
            raise ValueError(f"표지 {key}는 {limit}자 이하로 설정하세요.")
    return values


def _draw_cover(canvas, document, font_name, metadata):
    """A separate A4 title page; body headings and evidence stay unchanged."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph

    width, height = document.pagesize
    left = 25 * mm
    available = width - 2 * left
    ink = colors.HexColor("#172333")
    muted = colors.HexColor("#5B6674")

    def paragraph(text, top, size, leading, color=ink, max_height=40 * mm):
        style = ParagraphStyle(
            "Cover",
            fontName=font_name,
            fontSize=size,
            leading=leading,
            textColor=color,
            wordWrap="CJK",
        )
        item = Paragraph(_paragraph_text(text), style)
        _, item_height = item.wrap(available, max_height)
        if item_height > max_height:
            raise ValueError(
                "표지 정보가 지정된 공간을 초과합니다. 작성자·소속을 줄이세요."
            )
        item.drawOn(canvas, left, top - item_height)

    canvas.saveState()
    try:
        paragraph("TECHNICAL ASSESSMENT", height - 30 * mm, 10, 15, muted)
        canvas.setStrokeColor(ink)
        canvas.setLineWidth(0.8)
        canvas.line(left, height - 43 * mm, width - left, height - 43 * mm)
        paragraph("KV Cache 최적화 기술\n다관점 평가 보고서", height - 70 * mm, 25, 38)
        paragraph("KIVI와 InfiniGen 비교", height - 105 * mm, 15, 23, muted)
        paragraph(
            "GPU 기반 클라우드 LLM 서비스 적용 평가", height - 123 * mm, 11, 18, muted
        )
        paragraph(
            "기술 성숙도 / 시장성 / 이해관계자 / 도메인 적용성",
            height - 137 * mm,
            9,
            15,
            muted,
        )
        if metadata["authors"]:
            paragraph("작성자", 112 * mm, 9, 14, muted)
            paragraph(metadata["authors"], 102 * mm, 11, 18, max_height=35 * mm)
        if metadata["affiliation"]:
            paragraph(
                metadata["affiliation"], 62 * mm, 10, 16, muted, max_height=20 * mm
            )
        paragraph(
            "작성일  " + metadata["date"], 37 * mm, 10, 16, muted, max_height=15 * mm
        )
        canvas.setStrokeColor(colors.HexColor("#D5DCE3"))
        canvas.setLineWidth(0.5)
        canvas.line(left, 22 * mm, width - left, 22 * mm)
    finally:
        canvas.restoreState()


def _draw_body_footer(canvas, document, font_name):
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    canvas.saveState()
    try:
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#5B6674"))
        canvas.drawCentredString(
            document.pagesize[0] / 2, 10 * mm, str(document.page - 1)
        )
    finally:
        canvas.restoreState()


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
            Path(__file__).resolve().parents[4] / "assets" / "fonts" / "NanumGothic.ttf"
        )

    if not font_path.exists():
        raise FileNotFoundError(
            f"한글 PDF 생성을 위한 폰트 파일을 찾을 수 없습니다: {font_path}"
        )

    return font_path


def _write_pdf(
    report: ReportDraft,
    output_path: Path,
) -> None:
    """구조화된 보고서를 한글 PDF 파일로 저장한다."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    font_path = _resolve_font_path()
    font_name = "NanumGothic"
    metadata = _cover_metadata()

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
        author=metadata["authors"] or metadata["affiliation"],
    )

    styles = getSampleStyleSheet()

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

    # A non-empty first page triggers the cover callback; SUMMARY starts on page 2.
    story = [Spacer(1, 1), PageBreak()]

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

    document.build(
        story,
        onFirstPage=lambda canvas, doc: _draw_cover(canvas, doc, font_name, metadata),
        onLaterPages=lambda canvas, doc: _draw_body_footer(canvas, doc, font_name),
    )
