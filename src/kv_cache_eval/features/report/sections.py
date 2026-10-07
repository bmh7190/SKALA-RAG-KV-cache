"""작성·평가·수정 요청이 공유하는 보고서 대목차."""

from typing import Literal

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

ReportSection = Literal[REPORT_SECTION_TITLES[:-1]]
