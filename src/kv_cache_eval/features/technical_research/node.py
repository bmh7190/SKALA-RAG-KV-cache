"""기술별 조사 노드. 확인한 PDF 발췌만 State의 근거로 반환한다."""

import re

from kv_cache_eval.common.schemas import Evidence, ExperimentContext, Technology
from kv_cache_eval.common.state import State, StateUpdate


def _passage(text: str, match: re.Match[str], following: int = 0, preceding: int = 0) -> str:
    """일치한 표현이 포함된 문장과 필요한 경우 바로 다음 문장을 발췌한다."""
    boundary = text.rfind(". ", 0, match.start())
    start = 0 if boundary < 0 else boundary + 2
    for _ in range(preceding):
        boundary = text.rfind(". ", 0, max(0, start - 2))
        if boundary < 0 or start - boundary > 400:
            break
        start = boundary + 2
    if "Tuning-Free Asymmetric" in text[start:match.start()]:
        start = match.start()
    end = match.end()
    for _ in range(following + 1):
        boundary = text.find(". ", end)
        end = len(text) if boundary < 0 else boundary + 1
    return text[start:end].strip()


def _experiment(text: str) -> ExperimentContext | None:
    """발췌 chunk에 실제로 적힌 실험 식별자만 옮긴다."""
    context: ExperimentContext = {}
    models = list(dict.fromkeys(
        model.rstrip("-.") for model in re.findall(
            r"\b(?:Llama|LLaMA|Falcon|Mistral)-?\d[\w.-]*", text
        )
    ))
    if models:
        context["model"] = ", ".join(models[:4])
    workloads = list(dict.fromkeys(re.findall(
        r"\b(?:GSM8K|LongBench|RULER|CoQA|TruthfulQA|Wikitext-2|passkey retrieval)\b",
        text, flags=re.IGNORECASE,
    )))
    if workloads:
        context["workload"] = ", ".join(workloads[:3])
    baseline = re.search(r"\b(?:FP16|fp16|16bit|full precision) baseline\b", text)
    if baseline:
        context["baseline"] = baseline.group()
    return context or None


def research_technology(state: State, technology: Technology) -> StateUpdate:
    """KIVI 근거만 생성한다. InfiniGen은 담당 노드의 기존 상태를 유지한다."""
    if technology != "KIVI":
        raise NotImplementedError(f"{technology} 기술 조사 노드를 구현해야 합니다")

    # rag는 선택 의존성이므로 KIVI 실행 시에만 불러온다.
    from kv_cache_eval.features.technical_research.rag import build_index, retrieve

    index = build_index(include_validation=True)
    specs = (
        ("problem", "What KV cache memory bottleneck does KIVI address?", r"bottleneck in speed and memory usage", "primary", 0, 0),
        ("principle", "Why does KIVI quantize keys per channel and values per token?", r"key cache should be quantized per-channel", "primary", 1, 0),
        ("principle", "KIVI key cache per-channel and value cache per-token quantization", r"key cache should be quantized per-channel and value cache should be quantized per-token", "primary", 0, 0),
        ("mechanism", "How does KIVI use grouped quantization and full precision residual cache?", r"grouped key cache and value cache, while the residual", "primary", 1, 0),
        ("mechanism", "How does KIVI use a full precision residual window?", r"KIVI maintains a full precision KV cache sliding window", "primary", 1, 0),
        ("mechanism", "How does KIVI provide tuning-free 2bit KV cache quantization?", r"tuning-free 2bit KV cache quantization algorithm", "primary", 0, 0),
        ("experiment_conditions", "What group size and residual length are used in KIVI experiments?", r"group size G in Algorithm 1", "primary", 0, 0),
        ("experiment_conditions", "KIVI evaluation tasks CoQA TruthfulQA GSM8K LongBench normal and long context", r"adopt generation tasks from LM-Eval", "primary", 0, 0),
        ("experiment_conditions", "What GPU model baseline and workload are used for KIVI efficiency experiments?", r"single NVIDIA A100 GPU", "primary", 1, 0),
        ("performance_results", "What peak memory reduction does KIVI report?", r"less peak memory", "primary", 0, 0),
        ("performance_results", "What throughput improvement and batch size does KIVI report against FP16?", r"larger throughput", "primary", 0, 0),
        ("model_quality", "What accuracy drop does KIVI report on GSM8K?", r"accuracy drop is only around", "primary", 0, 0),
        ("model_quality", "What model quality or accuracy drop does KIVI report for Falcon and Mistral?", r"KIVI may have a large accuracy drop", "primary", 0, 1),
        ("limitations", "How does KIVI group size affect accuracy on hard tasks?", r"performance significantly decreases when the group size", "primary", 0, 0),
        ("limitations", "What implementation quantization overhead remains for KIVI?", r"overhead of quantization process", "primary", 0, 0),
        ("independent_evaluation", "How does KVQuant use KIVI as a comparison on long context retrieval?", r"comparisons with KIVI for reference", "independent_validation", 0, 1),
        ("independent_evaluation", "How does KVQuant compare KIVI on LongBench?", r"LongBench evaluation for the LLaMA-2-7B-32K", "independent_validation", 1, 0),
        ("independent_evaluation", "How does KVQuant compare KIVI on RULER and LongBench?", r"evaluation of KVQuant and KIVI on the RULER", "independent_validation", 1, 0),
        ("independent_evaluation", "KIVI open-source code with LLaMA does not support grouped-query attention", r"does not support grouped-query attention", "independent_validation", 0, 0),
    )
    evidence: list[Evidence] = []
    notes: list[str] = []
    category_counts: dict[str, int] = {}
    for category, query, anchor, role, following, preceding in specs:
        for hit in retrieve(index, query, k=8, source_role=role):
            text = " ".join(hit.page_content.split())
            match = re.search(anchor, text, flags=re.IGNORECASE)
            if match is None:
                continue
            excerpt = _passage(text, match, following, preceding)
            context_text = excerpt
            if category == "experiment_conditions" and "A100" in excerpt:
                context_text = text[max(0, match.start() - 280):match.end()]
            category_counts[category] = category_counts.get(category, 0) + 1
            evidence_id = f"kivi:{hit.metadata['source_id']}:{category}:{category_counts[category]}"
            item: Evidence = {
                "id": evidence_id,
                "technology": "KIVI",
                "claim": excerpt,
                "excerpt": excerpt,
                "source": {"document": hit.metadata["document"], "url": None, "page": hit.metadata["page"]},
                "experiment": _experiment(context_text),
                "limitations": [excerpt] if category in ("limitations", "model_quality") or (category == "independent_evaluation" and "does not support" in excerpt) else [],
                "verification_status": "source_checked",
            }
            evidence.append(item)
            notes.append(f"{evidence_id}: source_role={role}; source_technology={hit.metadata['source_technology']}")
            break
        else:
            notes.append(f"{category}: {role} 문서의 검색 chunk에서 직접 확인 가능한 구절을 찾지 못함")
    return {"kivi_evidence": {"evidence": evidence, "notes": notes}}


def research_kivi(state: State) -> StateUpdate:
    return research_technology(state, "KIVI")


def research_infinigen(state: State) -> StateUpdate:
    return research_technology(state, "InfiniGen")
