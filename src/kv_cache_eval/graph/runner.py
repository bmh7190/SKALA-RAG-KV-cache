"""환경·State 초기화 → 체크포인트 확인 → 그래프 실행 → 실행 결과 저장."""

import json
from pathlib import Path

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.state import State, new_state
from kv_cache_eval.graph.workflow import build_graph


def run(
    question: str,
    *,
    run_id: str | None = None,
    resume: bool = False,
    checkpoint_path: str | Path = "data/cache/supervisor/checkpoints.sqlite",
    overrides=None,
    interrupt_after=None,
    **limits,
) -> State:
    """Use the same run_id to resume an interrupted durable graph execution."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    load_environment()
    if resume and not run_id:
        raise ValueError("재개할 run_id가 필요합니다")
    state = new_state(question=question, trace_id=run_id, **limits)
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = path.parent / state["trace_id"]
    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph = build_graph(
            overrides,
            checkpointer=saver,
            interrupt_after=interrupt_after,
            trace_dir=trace_dir,
        )
        config = {
            "configurable": {"thread_id": state["trace_id"]},
            "metadata": {"trace_id": state["trace_id"]},
            "tags": ["supervisor", "main-ver2"],
            "recursion_limit": state["max_steps"] * 2 + 10,
        }
        snapshot = graph.get_state(config)
        if resume:
            if not snapshot.values:
                raise ValueError("재개할 체크포인트가 없습니다")
            config["recursion_limit"] = snapshot.values["max_steps"] * 2 + 10
            if not snapshot.next:
                return snapshot.values
        elif snapshot.values:
            raise ValueError("같은 run_id가 존재합니다. resume=True로 재개하세요")
        final = graph.invoke(None if resume else state, config=config)
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "state.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (trace_dir / "graph.mmd").write_text(
            graph.get_graph().draw_mermaid(), encoding="utf-8"
        )
        return final
