"""InfiniGen 원문 색인, 페이지 단위 검색 평가, 단독 Agentic RAG 실행."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from kv_cache_eval.common.config import load_environment
from kv_cache_eval.common.state import new_state
from kv_cache_eval.features.technical_research.infinigen.node import make_runtime_llm
from kv_cache_eval.features.technical_research.infinigen.prompts import questions_for_state
from kv_cache_eval.features.technical_research.infinigen.retriever import (
    RetrievalSettings, ensure_index,
)
from kv_cache_eval.features.technical_research.infinigen.workflow import ModelReviewer, research_questions


DEFAULT_FIXTURE = Path("tests/fixtures/infinigen_questions.json")
DEFAULT_CACHE = Path("data/cache/infinigen")


def evaluate_retrieval(retriever, fixture: dict, top_k: int) -> dict:
    answerable = [item for item in fixture["questions"] if item["answerable"]]
    negatives = [item for item in fixture["questions"] if not item["answerable"]]
    rows = []
    reciprocal_sum = 0.0
    hits_count = 0
    for item in answerable:
        expected = {(source["document_id"], page)
                    for source in item["relevant_sources"] for page in source["pdf_pages"]}
        hits = retriever.search(item["question"], top_k)
        rank = next((rank for rank, hit in enumerate(hits, 1)
                     if (hit.metadata["source_id"], hit.metadata["page"]) in expected), None)
        if rank is not None:
            hits_count += 1
            reciprocal_sum += 1 / rank
        rows.append({
            "id": item["id"], "question": item["question"], "expected": sorted(expected),
            "first_relevant_rank": rank,
            "retrieved": [{"source_id": hit.metadata["source_id"], "page": hit.metadata["page"],
                           "chunk_id": hit.metadata["chunk_id"], "rank": rank}
                          for rank, hit in enumerate(hits, 1)],
        })
    # 근거 없음 질문은 검색 결과만으로 정답/거절을 판정할 수 없다.
    for item in negatives:
        hits = retriever.search(item["question"], top_k)
        rows.append({"id": item["id"], "question": item["question"], "answerable": False,
                     "abstention_scored": False,
                     "retrieved": [{"source_id": hit.metadata["source_id"], "page": hit.metadata["page"],
                                    "chunk_id": hit.metadata["chunk_id"], "rank": rank}
                                   for rank, hit in enumerate(hits, 1)]})
    return {"question_count": len(answerable), "negative_count": len(negatives),
            "top_k": top_k, "hit_rate_at_k": hits_count / len(answerable),
            "mrr_at_k": reciprocal_sum / len(answerable), "questions": rows}


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--chunk-tokens", type=int, default=384)
    common.add_argument("--overlap-tokens", type=int, default=64)
    common.add_argument("--batch-size", type=int, default=2)
    common.add_argument("--device", default="cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", parents=[common], help="원문 검증, 색인 생성 또는 재사용")
    evaluation = commands.add_parser("evaluate", parents=[common], help="고정 질문집 검색 평가")
    evaluation.add_argument("--top-k", type=int, default=5)
    research = commands.add_parser("research", parents=[common], help="LLM을 이용한 단독 조사")
    research.add_argument("--top-k", type=int, default=5)
    research.add_argument("--max-attempts", type=int, default=2)
    research.add_argument("--max-llm-calls", type=int, default=18)
    research.add_argument("--questions", type=int, default=6, help="앞에서부터 조사할 질문 수, 1~6")
    args = parser.parse_args()
    load_environment()
    settings = RetrievalSettings(chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens,
                                 batch_size=args.batch_size, device=args.device)
    if args.command == "research":
        if not 1 <= args.questions <= 6:
            parser.error("--questions는 1~6이어야 합니다")
        llm = make_runtime_llm()  # 설정이 없으면 무거운 색인 생성보다 먼저 오류를 낸다.
    retriever, index_info = ensure_index(settings=settings)
    print(json.dumps({"index": index_info}, ensure_ascii=False))
    if args.command == "index":
        return
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        fixture_bytes = DEFAULT_FIXTURE.read_bytes()
        fixture = json.loads(fixture_bytes)
        result = evaluate_retrieval(retriever, fixture, args.top_k)
        result["provenance"] = {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
            "embedding_model": settings.resolved().model_name,
            "index_fingerprint": index_info["fingerprint"],
            "chunk_tokens": args.chunk_tokens, "overlap_tokens": args.overlap_tokens,
            "metric_scope": "12 fixed development questions, exact source ID + physical PDF page; negatives not scored",
        }
        target = DEFAULT_CACHE / "retrieval_eval.json"
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"hit_rate_at_k": result["hit_rate_at_k"], "mrr_at_k": result["mrr_at_k"],
                          "question_count": result["question_count"], "result_file": str(target)}, ensure_ascii=False))
        return
    reviewer = ModelReviewer(llm, args.max_llm_calls, target="InfiniGen")
    questions = questions_for_state(new_state(), target="InfiniGen")[:args.questions]
    result = research_questions(questions, retriever, reviewer, target="InfiniGen",
                                top_k=args.top_k, max_attempts=args.max_attempts)
    target = DEFAULT_CACHE / "research_result.json"
    target.write_text(json.dumps({
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.resolved().model_name,
        "llm_provider": os.environ["LLM_PROVIDER"],
        "llm_model": os.environ["LLM_MODEL"],
        "question_strategy": "selected-target-common-goals-v1",
        "index_fingerprint": index_info["fingerprint"],
        "llm_calls": reviewer.calls, "llm_call_limit": args.max_llm_calls,
        "questions": [item.id for item in questions], "result": result,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"evidence_count": len(result["evidence"]), "notes": len(result["notes"]),
                      "llm_calls": reviewer.calls, "result_file": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
