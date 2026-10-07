"""Explicit offline node doubles; they do not establish live model quality."""
from pathlib import Path
from kv_cache_eval.common.tasks import technologies, EVALUATION_KEYS
from kv_cache_eval.features.supervisor.catalog import CRITERIA
from kv_cache_eval.features.report.node import REPORT_SECTION_TITLES
from kv_cache_eval.features.quality.node import CRITERIA as QUALITY_CRITERIA


def evidence(tech, suffix='base'):
    return {'id':f'{tech}:{suffix}', 'technology':tech, 'claim':'검증용 원문', 'excerpt':'검증용 원문',
            'source':{'document':tech+'.pdf','url':None,'page':1},'experiment':None,
            'limitations':[], 'verification_status':'source_checked'}


def row(tech,criterion):
    return {'technology':tech,'criterion':criterion,'judgment':'검증용 판단','score':3,
            'rationale':'검증용 이유','evidence_ids':[f'{tech}:base'],'uncertainty':None,'basis_status':'inferred'}


def nodes(directory, events):
    def research(s):
        events.append('technical_research')
        return {('kivi_evidence' if tech=='KIVI' else 'infinigen_evidence'):
                {'evidence':[evidence(tech)],'notes':[]} for tech in technologies(s)}
    def evaluation(agent):
        def call(s):
            events.append(agent)
            return {EVALUATION_KEYS[agent]: {'evaluations':[row(t,c) for t in s['selected_technologies'] for c in CRITERIA[agent]], 'notes':[]}}
        return call
    def synthesize(s):
        events.append('synthesis')
        return {'synthesis':{'perspective_differences':['검증용'],'tradeoffs':['검증용'],
                            'application_conditions':['검증용'],'unresolved_gaps':s['evidence_gaps'],'cited_evidence_ids':['KIVI:base']}}
    def report(s):
        events.append('report')
        return {'report':{'sections':[(title,'' if title=='REFERENCE' else '오프라인 검증용 [KIVI:base] [InfiniGen:base]') for title in REPORT_SECTION_TITLES],
                          'cited_evidence_ids':['KIVI:base','InfiniGen:base']}}
    def quality(s):
        events.append('quality')
        return {'quality_result':{'report_revision':s['report_revision'],'passed':True,
                    'checks':{name:'pass' for name in QUALITY_CRITERIA}, 'reasons':{name:'테스트 대역' for name in QUALITY_CRITERIA}, 'issues':[]}}
    def export(s):
        events.append('export_pdf')
        from kv_cache_eval.features.report.node import export_pdf
        from unittest.mock import patch
        with patch.dict('os.environ',{'REPORT_PDF_PATH':str(Path(directory)/'report.pdf')}):
            return export_pdf(s)
    return {'technical_research':research,**{a:evaluation(a) for a in EVALUATION_KEYS},
            'synthesis':synthesize,'report':report,'quality':quality,'export_pdf':export}
