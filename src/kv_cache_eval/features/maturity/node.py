"""기술별 확인 근거가 보여 주는 검증 수준으로 공개 TRL을 추정한다."""

import re

from kv_cache_eval.common.schemas import Evidence, Evaluation, ResearchResult, Technology
from kv_cache_eval.common.state import State, StateUpdate


def _evaluate_one_technology(
    technology: Technology, research: ResearchResult | None,
) -> tuple[Evaluation, list[str]]:
    """기술명과 무관하게 동일한 TRL 1~9 기준을 적용한다."""
    independent_ids = set()
    if research is not None:
        independent_ids = {
            note.split(": source_role=", 1)[0]
            for note in research["notes"] if ": source_role=independent_validation" in note
        }
    verified = [] if research is None else [
        item for item in research["evidence"]
        if item["technology"] == technology
        and item["verification_status"] == "source_checked"
        and item["claim"].strip()
        and (item["source"]["document"].strip() or item["source"]["url"])
    ]

    def category(item: Evidence) -> str:
        parts = item["id"].split(":")
        return parts[-2].lower() if len(parts) >= 2 else ""

    direct = [
        item for item in verified
        if item["id"] not in independent_ids and category(item) != "independent_evaluation"
    ]

    def first(
        categories: set[str], terms: tuple[str, ...] = (), *,
        source: str | None = None, exclude_id: str | None = None,
    ) -> Evidence | None:
        return next((
            item for item in direct
            if category(item) in categories
            and (source is None or (item["source"]["document"] or item["source"]["url"]) == source)
            and item["id"] != exclude_id
            and (not terms or any(term in item["claim"].lower() for term in terms))
        ), None)

    concept = first({"principle", "concept"})
    method = first({"mechanism", "implementation"})
    experiment = first({"experiment", "model_quality", "performance_results", "component_validation"})
    model = next((item for item in direct if category(item) == "experiment_conditions"
                  and item["experiment"] and item["experiment"].get("model")), None)
    source = (model["source"]["document"] or model["source"]["url"]) if model else None
    workload = next((item for item in direct if category(item) == "experiment_conditions"
                     and item["experiment"] and item["experiment"].get("workload")
                     and source is not None
                     and (item["source"]["document"] or item["source"]["url"]) == source
                     and re.search(r"\b(real|realistic|representative|production-like|service|serving)\b",
                                   item["claim"], re.IGNORECASE)), None)
    implementation = first(
        {"implementation", "mechanism", "performance_results"},
        ("implement", "prototype", "system"), source=source,
    ) if source else None
    system_result = first(
        {"performance_results", "system_evaluation"},
        ("end-to-end", "wall-clock", "batch size", "peak memory", "latency", "system-level"),
        source=source, exclude_id=implementation["id"] if implementation else None,
    ) if source else None
    independent = next((item for item in verified if item["id"] in independent_ids
                        or category(item) == "independent_evaluation"), None)
    public_code = first({"implementation", "code", "publication"}, ("source code", "repository", "github"))

    score: float | None = None
    supporting = [item for item in (concept, method, experiment) if item]
    if concept:
        score = 1.0
        if method:
            score = 2.0
        if experiment:
            score = 3.0
        if method and experiment:
            score = 4.0
            if model and workload:
                score = 5.0
                supporting.extend((model, workload))
                if implementation and system_result:
                    score = 6.0
                    supporting.extend((implementation, system_result))

    operational = (
        (7, {"pilot", "operational_prototype"}, ("demonstrat", "시연", "실증")),
        (8, {"operational_validation"}, ("validat", "검증 완료")),
        (9, {"service_deployment", "production_deployment"}, ("deployed", "operat", "운용", "적용")),
    )
    if score is not None and score >= 6:
        for level, categories, terms in operational:
            item = first(categories, terms)
            if item and not re.search(r"\b(?:not|no|without)\b|미확인|않", item["claim"], re.IGNORECASE):
                score = float(level)
                supporting.append(item)

    if independent and score is not None:
        supporting.append(independent)  # 독립 검증은 점수를 올리는 조건이 아니다.
    evidence_ids = list(dict.fromkeys(item["id"] for item in supporting)) if score is not None else []
    rationale = None
    if score is not None:
        observations = ["기술 원리 또는 개념을 확인했다."]
        if method:
            observations.append("적용 방법 또는 핵심 구성 요소를 확인했다.")
        if experiment:
            observations.append("실험적 검증 결과를 확인했다.")
        if model and workload and score >= 5:
            observations.append("모델과 실제와 유사한 workload의 실험 조건을 확인했다.")
        if implementation and system_result and score >= 6:
            observations.append("구현 및 시스템 수준 결과를 함께 확인해 관련 환경의 시스템 시연으로 판단했다.")
        if independent:
            observations.append("독립 후속 검증은 보조 근거로만 사용했다.")
        if score >= 7:
            observations.append("해당 운영 단계의 명시적 공개 근거를 확인했다.")
        rationale = f"공개 자료 기반 TRL {int(score)} 추정이다. " + " ".join(observations)
        rationale += f" 사용한 Evidence ID: {', '.join(evidence_ids)}."

    unknown_from = int(score) + 1 if score is not None else 1
    uncertainty = "공개 코드 또는 독립 검증만으로 운영·서비스 적용을 판단하지 않는다. "
    if unknown_from <= 9:
        uncertainty += f"공개 Evidence에서 TRL {unknown_from}~9 도달 여부는 미확인이다. 미확인은 단계 미달을 뜻하지 않는다."
    evaluation: Evaluation = {
        "technology": technology,
        "criterion": "기술 성숙도(TRL)",
        "judgment": f"TRL {int(score)} (공개 자료 기반 추정)" if score is not None else "공개 자료로 판단 불가",
        "score": score,
        "rationale": rationale,
        "evidence_ids": evidence_ids,
        "uncertainty": uncertainty,
        "basis_status": "public_estimate" if score is not None else "unverified",
    }
    notes = [
        f"{technology}: 기술 개념 {'확인' if concept else '미확인'}, 구성 요소 {'확인' if method else '미확인'}, 실험 {'확인' if experiment else '미확인'}",
        f"{technology}: 관련 환경 {'확인' if model and workload else '미확인'}, 시스템 시연 {'확인' if implementation and system_result and score is not None and score >= 6 else '미확인'}",
        f"{technology}: 공개 코드 {'언급 확인' if public_code else '근거 미확인'}, 독립 검증 {'확인' if independent else '미확인'} (둘 다 운영 단계 근거로 승격하지 않음)",
        f"{technology}: TRL {unknown_from}~9 도달 여부 미확인 (단계 미달 판정 아님)" if unknown_from <= 9 else f"{technology}: TRL 9 공개 근거 확인",
    ]
    return evaluation, notes


def evaluate(state: State) -> StateUpdate:
    """현재 State의 기술별 조사 결과를 공통 TRL 평가로 연결한다."""
    research_by_technology = {
        "KIVI": state["kivi_evidence"],
        "InfiniGen": state["infinigen_evidence"],
    }
    evaluations: list[Evaluation] = []
    notes: list[str] = []
    for technology in state["selected_technologies"]:
        evaluation, technology_notes = _evaluate_one_technology(
            technology, research_by_technology[technology],
        )
        evaluations.append(evaluation)
        notes.extend(technology_notes)
    return {"maturity_eval": {"evaluations": evaluations, "notes": notes}}
