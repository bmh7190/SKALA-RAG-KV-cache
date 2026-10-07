"""품질 지적의 대상 장 정규화와 재작성 전후의 본문 비교."""

import re

from kv_cache_eval.common.errors import AgentFailure
from kv_cache_eval.common.evidence import CITATION
from kv_cache_eval.features.report.sections import REPORT_SECTION_TITLES


class ReportValidationError(AgentFailure):
    """코드가 작성한 안전한 보고서 수정 안내만 공개한다."""

    expose_message = True


def plain_text(text):
    """인용 표기·공백 차이가 본문 수정으로 취급되지 않도록 정규화한다."""
    return " ".join(CITATION.sub("", text).split())


def issue_sections(issues):
    """새 단일 장 ID와 기존 체크포인트의 복수 장·소제목 표현을 해석한다."""
    found = set()
    for issue in issues:
        section = issue["section"]
        if section == "전체":
            found.update(REPORT_SECTION_TITLES[:-1])
        else:
            found.update(t for t in REPORT_SECTION_TITLES[:-1] if t in section)
            if re.search(r"(?<!\d)4\.[1-4](?!\d)", section):
                found.add("4. 관점별 평가 결과")
    # 알 수 없는 기존 형식도 조용히 수정 대상을 잃지 않도록 전체 본문을 지정한다.
    return [t for t in REPORT_SECTION_TITLES[:-1] if t in found] or list(
        REPORT_SECTION_TITLES[:-1]
    )


def apply_revision(state, sections):
    """품질 수정 대상만 반영하고, 지적한 본문을 그대로 둔 재작성을 거부한다."""
    quality = state.get("quality_result")
    previous = state.get("report")
    if not previous or not quality or quality["passed"]:
        return sections
    if quality["report_revision"] != state["report_revision"] - 1:
        return sections
    issues = quality["issues"]
    targets = issue_sections(issues)
    old = dict(previous["sections"])
    updated = dict(sections)
    for issue in issues:
        quote = plain_text(issue.get("quote", ""))
        for title in issue_sections([issue]):
            if quote and quote in plain_text(updated[title]):
                raise ReportValidationError(
                    f"{title}: 지적된 문장이 그대로 남았습니다. 문제 주장을 삭제하거나 "
                    "대응 원문에 맞게 수정하고, 발췌의 한계를 논문 전체의 한계로 확대하지 마세요."
                )
            if not quote and plain_text(old.get(title, "")) == plain_text(
                updated[title]
            ):
                raise ReportValidationError(
                    f"{title}: 보완 지적이 본문에 반영되지 않았습니다. 필요한 판단·조건·한계를 보완하세요."
                )
    return [
        (title, text if title in targets or title not in old else old[title])
        for title, text in sections
    ]
