"""Supervisor graph using native conditional edges, persistence and tracing."""

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from kv_cache_eval.common.state import State, StateUpdate
from kv_cache_eval.features.domain.node import evaluate as evaluate_domain
from kv_cache_eval.features.market.node import evaluate as evaluate_market
from kv_cache_eval.features.maturity.node import evaluate as evaluate_maturity
from kv_cache_eval.features.quality.node import evaluate as evaluate_quality
from kv_cache_eval.features.report.node import export_pdf, write_report
from kv_cache_eval.features.stakeholders.node import evaluate as evaluate_stakeholders
from kv_cache_eval.features.supervisor.execution import worker
from kv_cache_eval.features.supervisor.node import supervise
from kv_cache_eval.features.synthesis.node import synthesize
from kv_cache_eval.features.technical_research.node import research
from kv_cache_eval.graph.validation import validate_input

Node = Callable[[State], StateUpdate]


def build_graph(
    overrides: Mapping[str, Node] | None = None,
    *,
    checkpointer=None,
    interrupt_after=None,
    trace_dir: Path | None = None,
):
    nodes = {
        "technical_research": research,
        "maturity": evaluate_maturity,
        "market": evaluate_market,
        "stakeholders": evaluate_stakeholders,
        "domain": evaluate_domain,
        "synthesis": synthesize,
        "report": write_report,
        "quality": evaluate_quality,
        "export_pdf": export_pdf,
    }
    if overrides:
        unknown = overrides.keys() - nodes.keys()
        if unknown:
            raise ValueError(f"알 수 없는 노드: {sorted(unknown)}")
        nodes.update(overrides)

    def supervisor(state):
        update = supervise(state)
        if trace_dir is not None:
            _record_decision(trace_dir, state, update)
        return {key: value for key, value in update.items() if state.get(key) != value}

    builder = StateGraph(State)
    builder.add_node("validate_input", validate_input)
    builder.add_node("supervisor", supervisor)
    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "supervisor")
    for name, function in nodes.items():
        builder.add_node(name, worker(name, function))
        builder.add_edge(name, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"] or END,
        {**{name: name for name in nodes}, END: END},
    )
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
        name="kv_cache_supervisor",
    )


def _record_decision(trace_dir, state, update):
    """라우팅 판단을 재현할 수 있도록 입력 결과와 배정 이유를 기록한다."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "trace_id": state["trace_id"],
        "step": update["step_count"],
        "next_agent": update["next_agent"],
        "decision": update["status"],
        "reason": update["decision_reason"],
        "request": update["retry_request"],
        "result": state.get("last_result"),
        "time": datetime.now(timezone.utc).isoformat(),
    }
    with (trace_dir / "decisions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
