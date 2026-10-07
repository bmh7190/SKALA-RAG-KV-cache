"""Supervisor graph using native conditional edges, persistence and tracing."""
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from kv_cache_eval.common.state import State, StateUpdate, new_state
from kv_cache_eval.common.config import load_environment
from kv_cache_eval.features.domain.node import evaluate as evaluate_domain
from kv_cache_eval.features.market.node import evaluate as evaluate_market
from kv_cache_eval.features.maturity.node import evaluate as evaluate_maturity
from kv_cache_eval.features.stakeholders.node import evaluate as evaluate_stakeholders
from kv_cache_eval.features.synthesis.node import synthesize
from kv_cache_eval.features.report.node import write_report, export_pdf
from kv_cache_eval.features.quality.node import evaluate as evaluate_quality
from kv_cache_eval.features.technical_research.node import research
from kv_cache_eval.features.supervisor.node import supervise
from kv_cache_eval.features.supervisor.execution import worker
from kv_cache_eval.graph.gates import validate_input

Node = Callable[[State], StateUpdate]


def build_graph(overrides: Mapping[str,Node] | None = None, *, user_question=None,
                checkpointer=None, interrupt_after=None, trace_dir: Path | None = None):
    nodes = {
        'technical_research': lambda state: research(state,user_question=state.get('question') or user_question),
        'maturity':evaluate_maturity, 'market':evaluate_market, 'stakeholders':evaluate_stakeholders,
        'domain':evaluate_domain, 'synthesis':synthesize, 'report':write_report,
        'quality':evaluate_quality, 'export_pdf':export_pdf,
    }
    if overrides:
        unknown = overrides.keys()-nodes.keys()
        if unknown: raise ValueError(f'알 수 없는 노드: {sorted(unknown)}')
        nodes.update(overrides)

    def supervisor(state):
        update = supervise(state)
        if trace_dir is not None:
            trace_dir.mkdir(parents=True,exist_ok=True)
            record = {'trace_id':state['trace_id'],'step':update['step_count'],
                      'next_agent':update['next_agent'],'decision':update['status'],
                      'reason':update['decision_reason'],'request':update['retry_request'],
                      'result':state.get('last_result'),'time':datetime.now(timezone.utc).isoformat()}
            with (trace_dir/'decisions.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        return {key: value for key, value in update.items() if state.get(key) != value}

    builder = StateGraph(State)
    builder.add_node('validate_input',validate_input)
    builder.add_node('supervisor',supervisor)
    builder.add_edge(START,'validate_input')
    builder.add_edge('validate_input','supervisor')
    for name, function in nodes.items():
        builder.add_node(name,worker(name,function))
        builder.add_edge(name,'supervisor')
    builder.add_conditional_edges('supervisor',lambda state: state['next_agent'] or END,
                                  {**{name:name for name in nodes},END:END})
    return builder.compile(checkpointer=checkpointer,interrupt_after=interrupt_after,name='kv_cache_supervisor')


def run(question: str, *, run_id: str | None = None, resume: bool = False,
        checkpoint_path: str | Path = 'data/cache/supervisor/checkpoints.sqlite',
        overrides=None, interrupt_after=None, **limits) -> State:
    """Use the same run_id to resume an interrupted durable graph execution."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    load_environment()
    if resume and not run_id: raise ValueError('재개할 run_id가 필요합니다')
    state = new_state(question=question,trace_id=run_id,**limits)
    path=Path(checkpoint_path); path.parent.mkdir(parents=True,exist_ok=True)
    trace_dir=path.parent/state['trace_id']
    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph=build_graph(overrides,user_question=question,checkpointer=saver,
                          interrupt_after=interrupt_after,trace_dir=trace_dir)
        config={'configurable':{'thread_id':state['trace_id']},
                'metadata':{'trace_id':state['trace_id']}, 'tags':['supervisor','main-ver2'],
                'recursion_limit':state['max_steps']*2+10}
        snapshot=graph.get_state(config)
        if resume:
            if not snapshot.values: raise ValueError('재개할 체크포인트가 없습니다')
            config['recursion_limit']=snapshot.values['max_steps']*2+10
            if not snapshot.next: return snapshot.values
        elif snapshot.values:
            raise ValueError('같은 run_id가 존재합니다. resume=True로 재개하세요')
        final=graph.invoke(None if resume else state,config=config)
        trace_dir.mkdir(parents=True,exist_ok=True)
        (trace_dir/'state.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
        (trace_dir/'graph.mmd').write_text(graph.get_graph().draw_mermaid(),encoding='utf-8')
        return final
