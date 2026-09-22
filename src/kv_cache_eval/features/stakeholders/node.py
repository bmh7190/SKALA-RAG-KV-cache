"""입력: 양쪽 조사 근거. 출력: stakeholder_eval 하나.

RAG-Design 설계서 3.4 "이해관계자 관점"의 5개 그룹(서비스·인프라 운영자, 모델·서빙
개발자, 서비스 이용 기업·사용자, 경쟁 기술 진영, 투자·산업 관계자)에 대해 KIVI/InfiniGen
각각을 평가한다.

도메인 평가 노드와 마찬가지로 이 노드는 스스로 문서를 검색하지 않는다 (설계서 2.1: 이해관계자
에이전트 RAG = X). technical_research가 만든 kivi_evidence / infinigen_evidence만 사용해
판단하고, 근거가 없거나 부족한 (기술, 이해관계자) 조합은 judgment/score를 None으로 남겨
evidence_gaps로 잡히게 한다 — "자료가 없다는 이유만으로 낮은 점수를 주지 않는다"는 설계서
원칙을, 아예 점수를 안 매기는 방식으로 지킨다.

이 노드는 기술 하나의 LLM 호출 실패(네트워크 오류, 검증 오류 등)가 다른 기술의 평가까지
망가뜨리지 않도록 기술 단위로 격리해서 처리한다.
"""

import os
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from kv_cache_eval.common.schemas import Evaluation, Evidence, Technology
from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.features.stakeholders.criteria import STAKEHOLDER_CRITERIA

_BasisStatus = Literal["unverified", "source_checked", "inferred", "public_estimate"]
_STAKEHOLDER_BY_GROUP = {c.group: c for c in STAKEHOLDER_CRITERIA}
_KNOWN_GROUPS = tuple(_STAKEHOLDER_BY_GROUP)


class _StakeholderAssessment(BaseModel):
    # 그룹명을 5개 고정값으로 제한한다 — LLM이 이름을 살짝 다르게(공백, 어순 등) 돌려주면
    # "새 그룹"으로 잘못 인식돼 항목이 중복 생성되는 것을 막기 위함. criterion은 LLM에게
    # 받지 않고, 검증된 stakeholder_group으로 STAKEHOLDER_CRITERIA에서 직접 채운다.
    stakeholder_group: Literal[_KNOWN_GROUPS]
    judgment: str | None = None
    score: float | None = Field(default=None, ge=1, le=5)
    rationale: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty: str | None = None
    basis_status: _BasisStatus = "unverified"


class _StakeholderAssessmentBatch(BaseModel):
    assessments: list[_StakeholderAssessment]
    notes: list[str] = Field(default_factory=list)


class SupportsStakeholderLLMCall(Protocol):
    def __call__(self, *, technology: Technology, evidence: list[Evidence]) -> _StakeholderAssessmentBatch: ...


def _default_llm_call(*, technology: Technology, evidence: list[Evidence]) -> _StakeholderAssessmentBatch:
    """환경변수로 지정된 OpenAI 모델을 호출한다 (팀 기본값: LLM_MODEL=gpt-5.4-mini).

    테스트에서는 이 함수를 쓰지 않고 evaluate()에 가짜 llm_call을 주입한다 (langchain-openai
    설치나 API 키가 없어도 노드 로직을 검증할 수 있도록). langchain-openai는 base 설치에는
    없는 선택 의존성(extra: llm-openai)이라 여기서만 지연 import한다.
    """
    from langchain_openai import ChatOpenAI

    from kv_cache_eval.common.config import load_environment

    load_environment()  # 이미 설정된 셸 값은 유지하고, .env가 있으면 채워 넣는다.

    model_name = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-5.4-mini"
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(_StakeholderAssessmentBatch)

    criteria_text = "\n".join(
        f"- {c.group}: {c.question}\n  점수 기준: {c.rubric}\n  확인할 근거 유형: {c.evidence_hint}"
        for c in STAKEHOLDER_CRITERIA
    )
    evidence_text = "\n".join(
        f"- id={e['id']} | claim={e['claim']} | excerpt={e['excerpt']} | "
        f"experiment={e['experiment']} | source={e['source']} | limitations={e['limitations']}"
        for e in evidence
    ) or "(제공된 근거 없음)"

    prompt = f"""당신은 KV cache 최적화 기술을 이해관계자 관점에서 평가하는 평가자입니다.
평가 대상 기술: {technology}

기술의 우열을 판정하지 말고, 각 이해관계자 입장에서 이 기술이 어떻게 평가될지만 기록하세요.
stakeholder_group은 아래 목록에 있는 이름 그대로만 써야 합니다(다른 표현으로 바꾸지 마세요):
{", ".join(_KNOWN_GROUPS)}

아래는 기술조사 담당이 원문에서 직접 확인(source_checked)한 {technology} 관련 근거입니다.
이 목록 밖의 지식이나 실제 확인되지 않은 반응을 지어내지 마세요. 근거로 뒷받침되지 않는
이해관계자 항목은 judgment와 score를 null로, evidence_ids를 빈 리스트로, basis_status를
"unverified"로 남기세요 ("미확인" — 근거가 없다는 이유로 낮은 점수를 주지 않습니다).

[근거 목록]
{evidence_text}

[이해관계자별 평가 기준]
{criteria_text}

정확히 {len(STAKEHOLDER_CRITERIA)}개 이해관계자 각각에 대해 하나씩 평가 항목을 반환하세요
(총 {len(STAKEHOLDER_CRITERIA)}개). evidence_ids에는 위 근거 목록에 실제로 존재하는 id만
쓸 수 있습니다. 이해관계자가 원문에서 직접 밝힌 반응이면 basis_status="source_checked",
다른 근거로부터 유추한 영향이면 "inferred"로 표시하세요. 경쟁 기술 진영 항목은 특히
경쟁사의 직접 반응과 연구자가 수행한 기술 비교를 구분해서 표시하세요.
"""
    return llm.invoke(prompt)


def _evaluate_technology(
    technology: Technology,
    evidence: list[Evidence],
    llm_call: SupportsStakeholderLLMCall,
    notes: list[str],
) -> list[Evaluation]:
    verified_evidence = [e for e in evidence if e["verification_status"] == "source_checked"]
    known_ids = {e["id"] for e in verified_evidence}

    if not verified_evidence:
        notes.append(f"{technology}: 확인된(source_checked) 근거가 없어 이해관계자 평가를 보류함")
        batch = _StakeholderAssessmentBatch(assessments=[], notes=[])
    else:
        try:
            batch = llm_call(technology=technology, evidence=verified_evidence)
        except Exception as exc:  # LLM 호출 실패가 다른 기술 평가까지 망가뜨리지 않도록 격리
            notes.append(f"{technology}: LLM 호출 실패로 전체 판단 보류 처리 ({type(exc).__name__}: {exc})")
            batch = _StakeholderAssessmentBatch(assessments=[], notes=[])

    notes.extend(batch.notes)

    evaluations: list[Evaluation] = []
    seen_groups: set[str] = set()
    for item in batch.assessments:
        seen_groups.add(item.stakeholder_group)
        criterion = _STAKEHOLDER_BY_GROUP[item.stakeholder_group].criterion

        evidence_ids = [eid for eid in item.evidence_ids if eid in known_ids]
        dropped = set(item.evidence_ids) - set(evidence_ids)
        if dropped:
            notes.append(f"{technology}/{item.stakeholder_group}: 확인되지 않은 근거 ID 제외됨 {sorted(dropped)}")

        judgment, score, basis_status = item.judgment, item.score, item.basis_status
        if not evidence_ids and (judgment is not None or score is not None):
            # 근거 없는 판단이 남지 않도록 보류(미확인) 상태로 되돌린다.
            judgment, score, basis_status = None, None, "unverified"

        evaluations.append(
            {
                "technology": technology,
                "criterion": criterion,
                "judgment": judgment,
                "score": score,
                "rationale": item.rationale,
                "evidence_ids": evidence_ids,
                "uncertainty": item.uncertainty,
                "basis_status": basis_status,
                "stakeholder_group": item.stakeholder_group,
            }
        )

    for stakeholder in STAKEHOLDER_CRITERIA:
        if stakeholder.group in seen_groups:
            continue
        notes.append(f"{technology}/{stakeholder.group}: 평가가 반환되지 않아 근거 부족으로 처리")
        evaluations.append(
            {
                "technology": technology,
                "criterion": stakeholder.criterion,
                "judgment": None,
                "score": None,
                "rationale": None,
                "evidence_ids": [],
                "uncertainty": "평가 응답 누락",
                "basis_status": "unverified",
                "stakeholder_group": stakeholder.group,
            }
        )

    return evaluations


def evaluate(state: State, llm_call: SupportsStakeholderLLMCall = _default_llm_call) -> StateUpdate:
    evaluations: list[Evaluation] = []
    notes: list[str] = []

    for technology, evidence_key in (("KIVI", "kivi_evidence"), ("InfiniGen", "infinigen_evidence")):
        research_result = state[evidence_key]
        if research_result is None:
            notes.append(f"{technology}: 기술조사 결과가 아직 없어 이해관계자 평가를 진행하지 못함")
            continue
        evaluations.extend(_evaluate_technology(technology, research_result["evidence"], llm_call, notes))

    return {"stakeholder_eval": {"evaluations": evaluations, "notes": notes}}
