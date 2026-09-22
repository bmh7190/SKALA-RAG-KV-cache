"""LangGraph 기본 배선. 기능 노드는 대체 가능하며 기본 구현은 미완료를 알린다."""

from collections.abc import Callable, Mapping

from langgraph.graph import END, START, StateGraph

from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.features.domain.node import evaluate as evaluate_domain
from kv_cache_eval.features.market.node import evaluate as evaluate_market
from kv_cache_eval.features.maturity.node import evaluate as evaluate_maturity
from kv_cache_eval.features.report.node import write_report
from kv_cache_eval.features.stakeholders.node import evaluate as evaluate_stakeholders
from kv_cache_eval.features.synthesis.node import synthesize
from kv_cache_eval.features.technical_research.node import research_infinigen, research_kivi
from kv_cache_eval.graph.gates import check_evidence, next_round, research_join, route_after_check, validate_input

Node = Callable[[State], StateUpdate]


def build_graph(overrides: Mapping[str, Node] | None = None):
    """그래프를 컴파일한다. 테스트는 기능 노드만 주입해 배선을 확인할 수 있다."""
    nodes: dict[str, Node] = {
        "validate_input": validate_input,
        "research_kivi": research_kivi,
        "research_infinigen": research_infinigen,
        "research_join": research_join,
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
    builder.add_edge("validate_input", "research_kivi")
    builder.add_edge("validate_input", "research_infinigen")
    builder.add_edge(["research_kivi", "research_infinigen"], "research_join")
    for name in ("maturity", "market", "stakeholders", "domain"):
        builder.add_edge("research_join", name)
    builder.add_edge(["maturity", "market", "stakeholders", "domain"], "evidence_check")
    builder.add_conditional_edges(
        "evidence_check",
        route_after_check,
        {"retry": "next_round", "synthesize": "synthesis"},
    )
    builder.add_edge("next_round", "research_kivi")
    builder.add_edge("next_round", "research_infinigen")
    builder.add_edge("synthesis", "report")
    builder.add_edge("report", END)
    return builder.compile()
