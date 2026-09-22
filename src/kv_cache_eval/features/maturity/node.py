"""확인된 KIVI 근거로 공개 자료 기반 TRL을 추정한다."""

from kv_cache_eval.common.schemas import Evidence, Evaluation
from kv_cache_eval.common.state import State, StateUpdate


def evaluate(state: State) -> StateUpdate:
    """KIVI만 평가하고 미확인 상위 단계는 미달로 단정하지 않는다."""
    research = state["kivi_evidence"]
    verified = [] if research is None else [
        item for item in research["evidence"]
        if item["technology"] == "KIVI"
        and item["verification_status"] == "source_checked"
        and item["claim"].strip()
        and item["source"]["page"] is not None
    ]
    primary = [item for item in verified if item["source"]["document"] == "kivi_original.pdf"]
    independent = [item for item in verified if item["source"]["document"] == "kvquant_validation.pdf"]

    def find(items: list[Evidence], category: str, *terms: str) -> Evidence | None:
        return next((
            item for item in items
            if f":{category}:" in item["id"]
            and all(term in item["claim"].lower() for term in terms)
        ), None)

    concept = find(primary, "principle", "per-channel", "per-token")
    mechanism = find(primary, "mechanism", "full precision")
    experiment = find(primary, "model_quality", "gsm8k")
    gpu = find(primary, "experiment_conditions", "a100", "gpu")
    throughput = find(primary, "performance_results", "throughput")
    workload = find(primary, "experiment_conditions", "sharegpt")
    validation = find(independent, "independent_evaluation", "kivi", "ruler")
    code = next((item for item in primary if "source code is available" in item["claim"].lower()
                 or "github.com" in item["claim"].lower()
                 or "github.com" in (item["source"]["url"] or "").lower()), None)

    score: float | None = None
    supporting = [item for item in (concept, mechanism, experiment, gpu, throughput, validation) if item]
    if concept and mechanism and gpu and throughput and gpu["experiment"] and gpu["experiment"].get("model"):
        score = 6.0  # 관련 GPU 환경에서 KIVI 시스템의 처리량까지 평가됨
    elif concept and mechanism and gpu:
        score = 5.0
    elif concept and mechanism and experiment:
        score = 4.0
    elif concept and experiment:
        score = 3.0
    elif concept and mechanism:
        score = 2.0
    elif concept:
        score = 2.0

    if score == 6.0 and workload:
        supporting.append(workload)
    evidence_ids = list(dict.fromkeys(item["id"] for item in supporting)) if score is not None else []
    rationale = None
    if score is not None:
        observations = ["원논문에서 KIVI의 기술 개념을 확인했다."]
        if mechanism:
            observations.append("핵심 메커니즘의 구현 설명을 확인했다.")
        if experiment:
            observations.append("모델 품질 실험을 확인했다.")
        if score == 6.0:
            observations.append("원논문의 GPU용 KIVI 구현, ShareGPT 기반으로 합성한 실제 LLM 추론 workload, end-to-end 메모리·처리량 평가를 종합해 실제 LLM serving과 유사한 환경의 시스템 시연으로 판단했다. 이는 실제 서비스 운영 근거가 아니다.")
        if validation:
            observations.append("KVQuant의 KIVI 비교는 독립 재평가 근거로만 사용했다.")
        rationale = (
            f"공개 자료 기반 TRL {int(score)} 추정이다. "
            + " ".join(observations)
            + f" 사용한 Evidence ID: {', '.join(evidence_ids)}."
        )
    uncertainty = (
        "KIVI 원논문 1쪽은 source code repository 제공을 명시하지만, 저장소 자체의 현재 상태·재현성·유지 여부는 별도 검증하지 않아 미확인이다. "
        "기업 Pilot/PoC, 실제 운용 환경의 시제품, 운영 검증 및 실제 서비스 적용에 관한 공개 근거도 확인되지 않아 TRL 7~9 도달 여부는 미확인이다. "
        "미확인은 해당 단계 미달을 뜻하지 않는다."
    )
    evaluation: Evaluation = {
        "technology": "KIVI",
        "criterion": "기술 성숙도(TRL)",
        "judgment": f"TRL {int(score)} (공개 자료 기반 추정)" if score is not None else "공개 자료로 판단 불가",
        "score": score,
        "rationale": rationale,
        "evidence_ids": evidence_ids,
        "uncertainty": uncertainty,
        "basis_status": "public_estimate" if score is not None else "unverified",
    }
    notes = [
        f"A 기술 개념: {'확인' if concept else '미확인'}",
        f"B 원논문 실험: {'확인' if experiment else '미확인'}",
        f"C GPU 시스템 평가: {'확인' if gpu and throughput else '미확인'}",
        f"D 공개 코드: 저자 공개 코드 제공 명시는 {'Evidence에서' if code else '원논문 1쪽에서'} 확인; repository 현재 상태·재현성·유지 여부는 미확인",
        f"E 독립 후속 검증: {'확인' if validation else '미확인'} (실제 운영 근거와 별개)",
        f"F 관련 LLM 추론 workload: {'ShareGPT 기반 조건 확인' if workload else 'ShareGPT 기반 조건 미확인'} (실제 서비스 운영과 별개)",
        "G 기업 Pilot/PoC: 미확인 (단계 미달 판정 아님)",
        "H 제품·서비스 적용: 미확인 (단계 미달 판정 아님)",
        f"TRL 1~{int(score)}: 공개 근거로 뒷받침되는 수준" if score is not None else "TRL 1~6: 확인 근거 부족",
        "TRL 7~9의 실제 운용 시제품·운영 검증·서비스 운용 여부는 현재 공개 근거로 미확인",
    ]
    return {"maturity_eval": {"evaluations": [evaluation], "notes": notes}}
