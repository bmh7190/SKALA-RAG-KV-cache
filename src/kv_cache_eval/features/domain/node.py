"""입력: 양쪽 조사 근거와 GPU 클라우드 LLM 도메인. 출력: domain_eval 하나.

RAG-Design 설계서 3.2 "도메인 적용 관점"의 6개 기준(GPU 메모리 사용량, 데이터 전송량,
추론 지연시간, 처리량, 모델 품질, 적용·운영 난이도)에 대해 KIVI/InfiniGen 각각을 평가한다.

이 노드는 스스로 문서를 검색하지 않는다. RAG는 technical_research 노드에서만 수행하기로
한 팀 설계(설계서 2.1)에 따라, kivi_evidence / infinigen_evidence에 이미 담긴 근거만
사용해 판단한다. 근거가 없거나 부족한 (기술, 기준) 조합은 judgment/score를 None으로
남겨 evidence_gaps로 잡히게 하고, 그 결과 재조사 라우팅(graph/gates.py)이나 synthesis의
unresolved_gaps로 이어지도록 한다 — 근거 없이 점수를 지어내지 않는다.
"""

import os
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from kv_cache_eval.common.schemas import Evaluation, Evidence, Technology
from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.features.domain.criteria import DOMAIN_CRITERIA

_BasisStatus = Literal["unverified", "source_checked", "inferred", "public_estimate"]


class _CriterionAssessment(BaseModel):
    criterion: str
    judgment: str | None = None
    score: float | None = Field(default=None, ge=1, le=5)
    rationale: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    uncertainty: str | None = None
    basis_status: _BasisStatus = "unverified"


class _DomainAssessmentBatch(BaseModel):
    assessments: list[_CriterionAssessment]
    notes: list[str] = Field(default_factory=list)


class SupportsDomainLLMCall(Protocol):
    def __call__(self, *, technology: Technology, domain: str, evidence: list[Evidence]) -> _DomainAssessmentBatch: ...


def _default_llm_call(*, technology: Technology, domain: str, evidence: list[Evidence]) -> _DomainAssessmentBatch:
    """환경변수로 지정된 OpenAI 모델을 호출한다 (팀 기본값: LLM_MODEL=gpt-5.4-mini).

    테스트에서는 이 함수를 쓰지 않고 evaluate()에 가짜 llm_call을 주입한다 (langchain-openai
    설치나 API 키가 없어도 노드 로직을 검증할 수 있도록). langchain-openai는 base 설치에는
    없는 선택 의존성(extra: llm-openai)이라 여기서만 지연 import한다.
    """
    from langchain_openai import ChatOpenAI

    from kv_cache_eval.common.config import load_environment

    load_environment()  # 이미 설정된 셸 값은 유지하고, .env가 있으면 채워 넣는다.

    model_name = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-5.4-mini"
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(_DomainAssessmentBatch)

    criteria_text = "\n".join(
        f"- {c.name}: {c.question}\n  점수 기준: {c.rubric}\n  확인할 근거 유형: {c.evidence_hint}"
        for c in DOMAIN_CRITERIA
    )
    evidence_text = "\n".join(
        f"- id={e['id']} | claim={e['claim']} | excerpt={e['excerpt']} | "
        f"experiment={e['experiment']} | source={e['source']} | limitations={e['limitations']}"
        for e in evidence
    ) or "(제공된 근거 없음)"

    prompt = f"""당신은 KV cache 최적화 기술의 도메인 적용 적합성을 평가하는 평가자입니다.
평가 도메인: {domain}
평가 대상 기술: {technology}

기술의 우열을 판정하지 말고, 이 도메인 관점에서 어떻게 평가되는지만 기록하세요.
아래는 기술조사 담당이 원문에서 직접 확인(source_checked)한 {technology} 관련 근거입니다.
이 목록 밖의 지식으로 점수를 만들어내지 마세요. 근거로 뒷받침되지 않는 기준은 judgment와
score를 null로, evidence_ids를 빈 리스트로, basis_status를 "unverified"로 남기세요.

[근거 목록]
{evidence_text}

[평가 기준]
{criteria_text}

정확히 {len(DOMAIN_CRITERIA)}개 기준 각각에 대해 하나씩 평가 항목을 반환하세요 (총 {len(DOMAIN_CRITERIA)}개).
evidence_ids에는 위 근거 목록에 실제로 존재하는 id만 쓸 수 있습니다. 근거가 해당 기준을 직접
측정한 실험 결과면 basis_status="source_checked", 다른 근거로부터 유추한 것이면 "inferred"로
표시하세요.
"""
    return llm.invoke(prompt)


def evaluate(state: State, llm_call: SupportsDomainLLMCall = _default_llm_call) -> StateUpdate:
    domain = state["domain_and_criteria"]["domain"]
    evaluations: list[Evaluation] = []
    notes: list[str] = []

    for technology, evidence_key in (("KIVI", "kivi_evidence"), ("InfiniGen", "infinigen_evidence")):
        research_result = state[evidence_key]
        if research_result is None:
            notes.append(f"{technology}: 기술조사 결과가 아직 없어 도메인 평가를 진행하지 못함")
            continue

        verified_evidence = [e for e in research_result["evidence"] if e["verification_status"] == "source_checked"]
        known_ids = {e["id"] for e in verified_evidence}

        batch = llm_call(technology=technology, domain=domain, evidence=verified_evidence)
        notes.extend(batch.notes)

        seen_criteria: set[str] = set()
        for item in batch.assessments:
            seen_criteria.add(item.criterion)
            evidence_ids = [eid for eid in item.evidence_ids if eid in known_ids]
            dropped = set(item.evidence_ids) - set(evidence_ids)
            if dropped:
                notes.append(f"{technology}/{item.criterion}: 확인되지 않은 근거 ID 제외됨 {sorted(dropped)}")

            judgment, score, basis_status = item.judgment, item.score, item.basis_status
            if not evidence_ids and (judgment is not None or score is not None):
                # 근거 없는 판단이 남지 않도록 보류 상태로 되돌린다 (근거 점검 게이트가 gap으로 잡는다).
                judgment, score, basis_status = None, None, "unverified"

            evaluations.append(
                {
                    "technology": technology,
                    "criterion": item.criterion,
                    "judgment": judgment,
                    "score": score,
                    "rationale": item.rationale,
                    "evidence_ids": evidence_ids,
                    "uncertainty": item.uncertainty,
                    "basis_status": basis_status,
                }
            )

        missing = {c.name for c in DOMAIN_CRITERIA} - seen_criteria
        for criterion_name in missing:
            notes.append(f"{technology}/{criterion_name}: 모델이 해당 기준을 반환하지 않아 근거 부족으로 처리")
            evaluations.append(
                {
                    "technology": technology,
                    "criterion": criterion_name,
                    "judgment": None,
                    "score": None,
                    "rationale": None,
                    "evidence_ids": [],
                    "uncertainty": "평가 응답 누락",
                    "basis_status": "unverified",
                }
            )

    return {"domain_eval": {"evaluations": evaluations, "notes": notes}}
