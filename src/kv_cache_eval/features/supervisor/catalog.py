"""One catalog for node ownership, evaluation scope, and request validation."""

from kv_cache_eval.common.tasks import EVALUATION_KEYS
from kv_cache_eval.features.domain.rubric import DOMAIN_RUBRIC
from kv_cache_eval.features.market.rubric import MARKET_CRITERIA
from kv_cache_eval.features.report.node import REPORT_SECTION_TITLES
from kv_cache_eval.features.stakeholders.criteria import STAKEHOLDER_CRITERIA
from kv_cache_eval.features.technical_research.prompts import QUESTION_TEMPLATES

CRITERIA = {
    "technical_research": tuple(q.id for q in QUESTION_TEMPLATES),
    "maturity": ("기술 성숙도(TRL)",),
    "market": MARKET_CRITERIA,
    "stakeholders": tuple(c.criterion for c in STAKEHOLDER_CRITERIA),
    "domain": tuple(DOMAIN_RUBRIC),
    "synthesis": (),
    "report": REPORT_SECTION_TITLES,
    "quality": (),
    "export_pdf": (),
}
OUTPUTS = {
    "technical_research": ("kivi_evidence", "infinigen_evidence"),
    **{name: (key,) for name, key in EVALUATION_KEYS.items()},
    "market": ("market_evidence", "market_eval"),
    "domain": ("domain_evidence", "domain_eval"),
    "synthesis": ("synthesis",),
    "report": ("report",),
    "quality": ("quality_result",),
    "export_pdf": ("pdf_path",),
}
WRITING = ("synthesis", "report", "quality", "export_pdf")
