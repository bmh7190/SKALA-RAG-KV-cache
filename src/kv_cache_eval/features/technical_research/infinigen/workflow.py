"""질문별 검색 -> 충분성 검토 -> 추출 / 제한된 재검색 LangGraph."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from kv_cache_eval.common.schemas import Evidence, ResearchResult
from kv_cache_eval.features.technical_research.infinigen.prompts import (
    EXTRACT_SYSTEM, REVIEW_SYSTEM, ResearchQuestion, hits_context,
)
from kv_cache_eval.features.technical_research.infinigen.retriever import LocalRetriever


class Review(BaseModel):
    sufficient: bool
    reason: str
    refinement_terms: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    chunk_id: str
    source_id: str
    page: int
    excerpt: str
    claim: str
    model: str | None = None
    workload: str | None = None
    baseline: str | None = None
    limitations: list[str] = Field(default_factory=list)


class Extraction(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class CallBudgetExhausted(RuntimeError):
    pass


class ModelReviewer:
    """실제 LLM 호출을 한 곳에서 계수하고 자동 재시도를 막는다."""

    def __init__(self, llm: object, max_calls: int, target: str):
        if max_calls <= 0:
            raise ValueError("max_calls는 양수여야 합니다")
        self.llm, self.max_calls, self.calls, self.target = llm, max_calls, 0, target

    def _invoke(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if self.calls >= self.max_calls:
            raise CallBudgetExhausted(f"LLM 호출 상한 {self.max_calls}회 소진")
        self.calls += 1
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"), ("human", "{user}"),
        ])
        return (prompt | self.llm.with_structured_output(schema)).invoke({
            "system": system, "user": user,
        })

    def review(self, question: str, hits: list[Document]) -> Review:
        return self._invoke(Review, REVIEW_SYSTEM,
                            f"Selected target: {self.target}\nQuestion: {question}\n\nRetrieved passages:\n{hits_context(hits)}")

    def extract(self, question: str, hits: list[Document]) -> Extraction:
        return self._invoke(Extraction, EXTRACT_SYSTEM,
                            f"Selected target: {self.target}\nQuestion: {question}\n\nRetrieved passages:\n{hits_context(hits)}")


class QuestionState(TypedDict):
    question: str
    query: str
    attempt: int
    max_attempts: int
    top_k: int
    hits: list[Document]
    review: Review | None
    evidence: list[Evidence]
    notes: list[str]


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ground_claim(claim: Claim, hits: list[Document], *, target: str) -> Evidence | None:
    """LLM의 출처 정보가 검색된 청크와 일치할 때만 source_checked로 기록."""
    doc = next((item for item in hits if item.metadata["chunk_id"] == claim.chunk_id), None)
    if doc is None:
        return None
    meta = doc.metadata
    excerpt = _normalized(claim.excerpt)
    if (claim.source_id != meta["source_id"] or claim.page != meta["page"]
            or not excerpt or excerpt not in _normalized(doc.page_content)
            or not claim.claim.strip()):
        return None
    if meta["source_role"] == "primary" and meta["subject_technology"].casefold() != target.casefold():
        return None
    if meta["source_role"] != "primary":
        # 비교 문서의 자체 결과를 조사 대상으로 둔갑시키지 않는다.
        if meta["subject_technology"].casefold() not in claim.claim.casefold() or target.casefold() in claim.claim.casefold():
            return None
    stable = f"{meta['source_id']}:{meta['page']}:{excerpt}".encode("utf-8")
    evidence_id = "infinigen-" + hashlib.sha256(stable).hexdigest()[:20]
    passage = _normalized(doc.page_content).casefold()
    context = {key: _normalized(value) for key, value in {
        "model": claim.model, "workload": claim.workload, "baseline": claim.baseline,
    }.items() if value and _normalized(value).casefold() in passage}
    return {
        "id": evidence_id, "technology": target, "claim": claim.claim.strip(),
        "excerpt": excerpt,
        "source": {"document": meta["source_id"], "url": meta["source_url"], "page": meta["page"]},
        "experiment": context or None,
        "limitations": claim.limitations,
        "verification_status": "source_checked",
    }


def build_question_graph(
    search: Callable[[str, int], list[Document]],
    review: Callable[[str, list[Document]], Review],
    extract: Callable[[str, list[Document]], Extraction],
    *, target: str,
):
    def draft(state: QuestionState) -> dict:
        return {"query": state["question"]}

    def retrieve(state: QuestionState) -> dict:
        return {"attempt": state["attempt"] + 1,
                "hits": search(state["query"], state["top_k"])}

    def grade(state: QuestionState) -> dict:
        if not state["hits"]:
            decision = Review(sufficient=False, reason="검색 결과 없음")
        else:
            decision = review(state["question"], state["hits"])
        return {"review": decision}

    def route_review(state: QuestionState) -> Literal["extract", "revise", "unresolved"]:
        if state["review"] and state["review"].sufficient:
            return "extract"
        return "revise" if state["attempt"] < state["max_attempts"] else "unresolved"

    def revise(state: QuestionState) -> dict:
        bodies = [_normalized(doc.page_content).casefold() for doc in state["hits"]]
        terms = []
        for candidate in (state["review"].refinement_terms if state["review"] else [])[:3]:
            term = _normalized(candidate)
            if 2 <= len(term) <= 80 and any(term.casefold() in body for body in bodies) and term not in terms:
                terms.append(term)
        # 원래 목적을 보존하고, 검증된 본문 용어만 검색 힌트로 추가한다.
        query = state["question"] + (" " + " ".join(terms) if terms else " 원문 근거 설명 측정 비교")
        return {"query": query}

    def extract_node(state: QuestionState) -> dict:
        raw = extract(state["question"], state["hits"])
        verified = [item for claim in raw.claims if (item := ground_claim(claim, state["hits"], target=target))]
        unique = {item["id"]: item for item in verified}
        notes = list(state["notes"])
        if len(verified) < len(raw.claims):
            notes.append("출처 ID·페이지·발췌 또는 비교 문서 귀속이 맞지 않는 후보 근거를 제외함")
        return {"evidence": list(unique.values()), "notes": notes}

    def route_extract(state: QuestionState) -> Literal["done", "revise", "unresolved"]:
        if state["evidence"]:
            return "done"
        return "revise" if state["attempt"] < state["max_attempts"] else "unresolved"

    def unresolved(state: QuestionState) -> dict:
        reason = state["review"].reason if state["review"] else "추출된 근거 없음"
        return {"notes": state["notes"] + [f"미확인: {state['question']} (시도 {state['attempt']}/{state['max_attempts']}; {reason})"]}

    builder = StateGraph(QuestionState)
    for name, func in (("draft", draft), ("retrieve", retrieve), ("grade", grade),
                       ("revise", revise), ("extract", extract_node), ("unresolved", unresolved)):
        builder.add_node(name, func)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_review,
                                  {"extract": "extract", "revise": "revise", "unresolved": "unresolved"})
    builder.add_edge("revise", "retrieve")
    builder.add_conditional_edges("extract", route_extract,
                                  {"done": END, "revise": "revise", "unresolved": "unresolved"})
    builder.add_edge("unresolved", END)
    return builder.compile()


def research_questions(
    questions: list[ResearchQuestion], retriever: LocalRetriever, reviewer: ModelReviewer,
    *, target: str, top_k: int = 5, max_attempts: int = 2, prior: ResearchResult | None = None,
) -> ResearchResult:
    if not 1 <= max_attempts <= 5 or top_k <= 0:
        raise ValueError("max_attempts는 1~5, top_k는 양수여야 합니다")
    graph = build_question_graph(retriever.search, reviewer.review, reviewer.extract, target=target)
    evidence = {item["id"]: item for item in (prior or {}).get("evidence", [])}
    notes = list((prior or {}).get("notes", []))
    for question_index, question in enumerate(questions):
        try:
            result = graph.invoke({
                "question": question.text, "query": "", "attempt": 0,
                "max_attempts": max_attempts, "top_k": top_k,
                "hits": [], "review": None, "evidence": [], "notes": [],
            }, config={"recursion_limit": max_attempts * 4 + 8})
        except CallBudgetExhausted as error:
            notes.append(str(error))
            notes.extend(f"미확인(호출 상한): {pending.id}: {pending.text}"
                         for pending in questions[question_index:])
            break
        for item in result["evidence"]:
            evidence[item["id"]] = item
        notes.extend(f"{question.id}: {note}" for note in result["notes"])
    return {"evidence": list(evidence.values()), "notes": notes}
