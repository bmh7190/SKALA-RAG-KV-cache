"""입력 확인, 구조적 근거 공백 확인, 제한된 재조사 라우팅."""

from typing import Literal

from kv_cache_eval.common.schemas import EvidenceGap, Technology
from kv_cache_eval.common.state import State, StateUpdate


def validate_input(state: State) -> StateUpdate:
    if state["selected_technologies"] != ("KIVI", "InfiniGen"):
        raise ValueError("현재 그래프는 KIVI와 InfiniGen 비교용입니다")
    if not state["domain_and_criteria"]["domain"]:
        raise ValueError("평가 도메인이 필요합니다")
    return {}


def check_evidence(state: State) -> StateUpdate:
    """최소 구조 검사. 출처의 실제 신뢰성·주장 타당성 검토는 TODO."""
    gaps: list[EvidenceGap] = []
    known_ids: dict[Technology, set[str]] = {"KIVI": set(), "InfiniGen": set()}
    all_ids: set[str] = set()
    for technology, key in (("KIVI", "kivi_evidence"), ("InfiniGen", "infinigen_evidence")):
        result = state[key]
        if result is None or not result["evidence"]:
            gaps.append({"technology": technology, "criterion": "기술 조사", "reason": "확인된 근거가 없음"})
            continue
        for evidence in result["evidence"]:
            source = evidence["source"]
            if evidence["id"] in all_ids:
                gaps.append({"technology": technology, "criterion": "기술 조사", "reason": f"중복 근거 ID: {evidence['id']}"})
            all_ids.add(evidence["id"])
            if (evidence["technology"] != technology
                    or evidence["verification_status"] != "source_checked"
                    or not evidence["claim"].strip()
                    or not (source["document"].strip() or source["url"])):
                gaps.append({"technology": technology, "criterion": "기술 조사", "reason": f"근거 {evidence['id']}의 기술/출처 확인 필요"})
            else:
                known_ids[technology].add(evidence["id"])

    market_result = state["market_evidence"]
    if market_result is not None:
        for evidence in market_result["evidence"]:
            technology = evidence["technology"]
            source = evidence["source"]
            if evidence["id"] in all_ids:
                gaps.append({"technology": technology, "criterion": "시장성", "reason": f"중복 근거 ID: {evidence['id']}"})
            all_ids.add(evidence["id"])
            if (evidence["verification_status"] != "source_checked"
                    or not evidence["claim"].strip()
                    or not (source["document"].strip() or source["url"])):
                gaps.append({"technology": technology, "criterion": "시장성", "reason": f"근거 {evidence['id']}의 출처 확인 필요"})
            else:
                known_ids[technology].add(evidence["id"])

    for criterion, key in (
        ("기술 성숙도(TRL)", "maturity_eval"),
        ("시장성", "market_eval"),
        ("이해관계자", "stakeholder_eval"),
        ("도메인 적합성", "domain_eval"),
    ):
        result = state[key]
        for technology in ("KIVI", "InfiniGen"):
            assessments = [] if result is None else [item for item in result["evaluations"] if item["technology"] == technology]
            if not assessments:
                gaps.append({"technology": technology, "criterion": criterion, "reason": "평가 결과가 없음"})
                continue
            for assessment in assessments:
                if ((assessment["judgment"] is None and assessment["score"] is None)
                        or assessment["basis_status"] == "unverified"
                        or not assessment["evidence_ids"]):
                    gaps.append({"technology": technology, "criterion": criterion, "reason": "판단 근거가 미확인"})
                elif any(evidence_id not in known_ids[technology] for evidence_id in assessment["evidence_ids"]):
                    gaps.append({"technology": technology, "criterion": criterion, "reason": "확인된 근거 ID와 연결되지 않음"})
    return {"evidence_gaps": gaps}


def route_after_check(state: State) -> Literal["retry", "synthesize"]:
    if state["evidence_gaps"] is None:
        raise ValueError("근거 확인 결과가 없습니다")
    if state["evidence_gaps"] and state["research_round"] < state["max_research_rounds"]:
        return "retry"
    # 횟수 소진 시에도 synthesis가 미해결 공백을 명시해야 한다.
    return "synthesize"


def next_round(state: State) -> StateUpdate:
    if not state["evidence_gaps"] or state["research_round"] >= state["max_research_rounds"]:
        raise ValueError("추가 조사를 시작할 조건이 아닙니다")
    return {"research_round": state["research_round"] + 1}
