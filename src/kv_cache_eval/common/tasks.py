"""Shared work scope and deterministic partial-result merging."""
from kv_cache_eval.common.evidence import collect_evidence

RESEARCH_KEYS = {'KIVI': 'kivi_evidence', 'InfiniGen': 'infinigen_evidence'}
EVALUATION_KEYS = {'maturity': 'maturity_eval', 'market': 'market_eval',
                   'stakeholders': 'stakeholder_eval', 'domain': 'domain_eval'}


class InputBudgetExceeded(ValueError):
    pass


def technologies(state):
    target = (state.get('retry_request') or {}).get('technology')
    if target is not None and target not in state['selected_technologies']:
        raise ValueError('알 수 없는 재작업 기술')
    return (target,) if target else state['selected_technologies']


def criteria(state, allowed):
    requested = (state.get('retry_request') or {}).get('criteria', [])
    if set(requested) - set(allowed):
        raise ValueError(f'알 수 없는 재작업 항목: {requested}')
    return requested or list(allowed)


def merge_evaluation(state, key, result):
    def identity(row):
        return row['technology'], row['criterion'], row.get('stakeholder_group')
    rows = {identity(row): row for row in (state.get(key) or {}).get('evaluations', [])}
    rows.update((identity(row), row) for row in result['evaluations'])
    merged = {**result, 'evaluations': list(rows.values())}
    # Generated text may describe only the requested subset; consumers use structured rows.
    merged.pop('text', None)
    return merged


def merge_research(previous, current):
    items = {item['id']: item for item in (previous or {}).get('evidence', [])}
    items.update((item['id'], item) for item in current['evidence'])
    return {'evidence': list(items.values()), 'notes': list(dict.fromkeys(current.get('notes', [])))}


def domain_references(state, evaluation):
    evidence = collect_evidence({**state, 'domain_evidence': None})
    ids = {eid for row in evaluation['evaluations'] for eid in row['evidence_ids']}
    return {'evidence': [evidence[eid] for eid in sorted(ids) if eid in evidence],
            'notes': ['도메인 평가가 실제 인용한 원래 근거와 ID를 유지함']}
