"""평가기의 주장과 실제 본문·근거 연결을 검증한다."""

from typing import Literal

from pydantic import BaseModel, Field, create_model

from kv_cache_eval.common.errors import AgentFailure
from kv_cache_eval.features.report.feedback import plain_text
from kv_cache_eval.features.report.sections import ReportSection


class QualityValidationError(AgentFailure):
    """잘못된 평가 응답은 보고서 수정 대신 평가기에 다시 요청한다."""

    expose_message = True


class Finding(BaseModel):
    section: ReportSection
    kind: Literal["claim", "missing_content"]
    quote: str
    evidence_ids: list[str]
    reason: str = Field(min_length=1)
    required_action: str = Field(min_length=1)


class Check(BaseModel):
    status: Literal["pass", "fail", "unknown"]
    reason: str = Field(min_length=1)
    findings: list[Finding]


class Judgment(BaseModel):
    groundedness: Check
    neutrality: Check
    bias_control: Check
    perspective_coverage: Check


def judgment_schema(evidence):
    """Judge도 작성기처럼 제공된 근거 ID만 구조화 출력에서 선택한다."""
    finding = create_model(
        "EvidenceFinding",
        __base__=Finding,
        evidence_ids=(list[Literal[tuple(evidence)]], ...),
    )
    check = create_model("EvidenceCheck", __base__=Check, findings=(list[finding], ...))
    return create_model(
        "EvidenceJudgment", **{name: (check, ...) for name in Judgment.model_fields}
    )


def validate_judgment(judgment, report, evidence):
    """허구 인용·존재하지 않는 장·통과 판정과 지적의 모순을 수락하지 않는다."""
    sections = dict(report["sections"])
    for criterion, check in judgment.items():
        findings = check["findings"]
        if not check["reason"].strip() or (check["status"] == "pass") != (not findings):
            raise QualityValidationError(
                "pass에는 지적을 넣지 않고 fail/unknown에는 구체적인 지적을 작성하세요."
            )
        for finding in findings:
            if not finding["reason"].strip() or not finding["required_action"].strip():
                raise QualityValidationError(
                    "문제 이유와 실행 가능한 수정 행동을 작성하세요."
                )
            quote = plain_text(finding["quote"])
            if finding["kind"] == "claim":
                if not quote or quote not in plain_text(
                    sections.get(finding["section"], "")
                ):
                    raise QualityValidationError(
                        "quote는 해당 section의 실제 본문을 그대로 인용하세요. 검사 규칙이나 원문을 보고서 문장으로 인용하지 마세요."
                    )
            elif criterion != "perspective_coverage" or quote:
                raise QualityValidationError(
                    "missing_content는 관점 누락에만 사용하고 quote는 빈 문자열로 작성하세요."
                )
            if set(finding["evidence_ids"]) - evidence.keys():
                raise QualityValidationError(
                    "evidence_ids는 제공된 근거 ID에서만 선택하세요."
                )
    return judgment
