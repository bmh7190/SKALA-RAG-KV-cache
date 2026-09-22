"""공통 기술 조사와 평가를 연결하는 LangGraph 실행 흐름."""

from collections.abc import Callable, Mapping

from langgraph.graph import END, START, StateGraph

from kv_cache_eval.common.state import State, StateUpdate, new_state
from kv_cache_eval.features.domain.node import evaluate as evaluate_domain
from kv_cache_eval.features.market.node import evaluate as evaluate_market
from kv_cache_eval.features.maturity.node import evaluate as evaluate_maturity
from kv_cache_eval.features.report.node import write_report
from kv_cache_eval.features.stakeholders.node import evaluate as evaluate_stakeholders
from kv_cache_eval.features.synthesis.node import synthesize
from kv_cache_eval.features.technical_research.node import research
from kv_cache_eval.graph.gates import check_evidence, next_round, route_after_check, validate_input

Node = Callable[[State], StateUpdate]


def build_graph(overrides: Mapping[str, Node] | None = None, *, user_question: str | None = None):
    """그래프를 컴파일한다. 테스트는 기능 노드만 주입해 배선을 확인할 수 있다."""
    nodes: dict[str, Node] = {
        "validate_input": validate_input,
        "technical_research": lambda state: research(state, user_question=user_question),
        "maturity": evaluate_maturity,
        "market": evaluate_market,
        "stakeholders": evaluate_stakeholders,
        "domain": evaluate_domain,
        "evidence_check": check_evidence,
        "next_round": next_round,
        "synthesis": synthesize,
        "report": write_report,
    }
    if overrides:
        unknown = set(overrides) - set(nodes)
        if unknown:
            raise ValueError(f"알 수 없는 노드: {sorted(unknown)}")
        nodes.update(overrides)

    builder = StateGraph(State)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "technical_research")
    for name in ("maturity", "market", "stakeholders", "domain"):
        builder.add_edge("technical_research", name)
    builder.add_edge(["maturity", "market", "stakeholders", "domain"], "evidence_check")
    builder.add_conditional_edges(
        "evidence_check",
        route_after_check,
        {"retry": "next_round", "synthesize": "synthesis"},
    )
    builder.add_edge("next_round", "technical_research")
    builder.add_edge("synthesis", "report")
    builder.add_edge("report", END)
    return builder.compile()


def run(question: str) -> State:
    """질문을 공통 조사 노드에 전달해 전체 Graph를 실행한다."""
    return build_graph(user_question=question).invoke(new_state())
