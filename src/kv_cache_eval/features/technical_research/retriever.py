"""LangChain HuggingFaceEmbeddings + FAISS retriever와 검증된 로컬 캐시."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kv_cache_eval.common.config import get_embedding_model
from kv_cache_eval.features.technical_research.ingest import (
    DEFAULT_DOCUMENT_DIR, Source, file_sha256, load_sources, make_chunks, validate_all_sources,
)

if TYPE_CHECKING:
    from langchain_core.documents import Document


INDEX_FORMAT = 3
INDEX_FILES = ("index.faiss", "index.pkl")


@dataclass(frozen=True)
class RetrievalSettings:
    model_name: str = ""
    chunk_tokens: int = 384
    overlap_tokens: int = 64
    batch_size: int = 2
    device: str = "cpu"

    def resolved(self) -> "RetrievalSettings":
        return RetrievalSettings(self.model_name or get_embedding_model(), self.chunk_tokens,
                                 self.overlap_tokens, self.batch_size, self.device)


def index_fingerprint(sources: list[Source], settings: RetrievalSettings) -> str:
    payload = {
        "format": INDEX_FORMAT,
        "sources": [
            (item.document_id, item.citation_document, item.title, item.source_url, item.version,
             item.evidence_role, item.subject_technology, item.file,
             item.sha256, item.page_count)
            for item in sources
        ],
        "model": settings.model_name,
        "chunk_tokens": settings.chunk_tokens,
        "overlap_tokens": settings.overlap_tokens,
        "extraction": "langchain-pypdf-page-recursive-hf-tokenizer-v3",
        "vector": "langchain-faiss-normalized-embedding-l2-v3",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def index_is_current(index_dir: Path, fingerprint: str) -> bool:
    """이 코드가 쓴 로컬 manifest와 파일 해시가 맞을 때만 캐시를 재사용한다."""
    try:
        meta = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        return (
            meta["format"] == INDEX_FORMAT and meta["fingerprint"] == fingerprint
            and meta["created_by"] == "kv-cache-eval:technical-research"
            and all(file_sha256(index_dir / name) == meta["files"][name]
                    for name in INDEX_FILES)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _embedder(settings: RetrievalSettings) -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.model_name,
        model_kwargs={"device": settings.device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.batch_size},
    )


class LocalRetriever:
    def __init__(self, vectorstore: Any):
        self.vectorstore = vectorstore

    def search(self, query: str, top_k: int = 5, roles: set[str] | None = None) -> list[Document]:
        if top_k <= 0:
            raise ValueError("top_k는 양수여야 합니다")
        options: dict[str, Any] = {"k": top_k}
        if roles is not None:
            options["filter"] = {"source_role": {"$in": sorted(roles)}}
            options["fetch_k"] = self.vectorstore.index.ntotal
        return self.vectorstore.as_retriever(search_kwargs=options).invoke(query)


def _read_index(index_dir: Path, embedder: Any, sources: list[Source]) -> LocalRetriever:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    meta = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    # index_is_current가 지문과 두 파일의 해시를 확인한 뒤에만 호출된다.
    store = FAISS.load_local(str(index_dir), embedder, allow_dangerous_deserialization=True)
    if store.index.d != meta["dimension"] or store.index.ntotal != meta["chunk_count"]:
        raise ValueError("저장된 색인의 차원 또는 청크 수가 manifest와 다릅니다")
    source_by_id = {source.document_id: source for source in sources}
    if len(store.index_to_docstore_id) != store.index.ntotal:
        raise ValueError("저장된 벡터와 문서 매핑 수가 다릅니다")
    for doc_id in store.index_to_docstore_id.values():
        doc = store.docstore.search(doc_id)
        if not isinstance(doc, Document):
            raise ValueError("저장된 문서를 읽을 수 없습니다")
        source = source_by_id.get(doc.metadata.get("source_id"))
        if (source is None or doc.id != doc.metadata.get("chunk_id")
                or doc.metadata.get("source_document") != source.citation_document
                or doc.metadata.get("source_role") != source.evidence_role
                or doc.metadata.get("source_url") != source.source_url
                or not 1 <= doc.metadata.get("page", 0) <= source.page_count):
            raise ValueError("저장된 문서의 출처·페이지 메타데이터가 다릅니다")
    vector = embedder.embed_query("dimension check")
    if len(vector) != store.index.d or not all(math.isfinite(value) for value in vector):
        raise ValueError("현재 임베딩 모델과 저장된 색인의 차원이 다릅니다")
    return LocalRetriever(store)


def ensure_index(
    technology: str = "InfiniGen",
    document_dir: Path = DEFAULT_DOCUMENT_DIR,
    index_dir: Path | None = None,
    settings: RetrievalSettings | None = None,
) -> tuple[LocalRetriever, dict[str, Any]]:
    """원문 검증 후 LangChain FAISS 캐시를 재사용하거나 새로 만든다."""
    import faiss
    import torch
    from langchain_community.vectorstores import FAISS
    from transformers import AutoTokenizer

    settings = (settings or RetrievalSettings()).resolved()
    if settings.batch_size <= 0 or settings.chunk_tokens <= 0 or not 0 <= settings.overlap_tokens < settings.chunk_tokens:
        raise ValueError("검색/청킹 설정값이 유효하지 않습니다")
    if settings.device == "cpu":
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
    sources, _ = load_sources(technology)
    counts = validate_all_sources(document_dir)
    pages = counts[technology]
    index_dir = index_dir or Path("data/indexes") / technology.lower()
    fingerprint = index_fingerprint(sources, settings)
    embedder = _embedder(settings)
    if index_is_current(index_dir, fingerprint):
        retriever = _read_index(index_dir, embedder, sources)
        return retriever, {"status": "reused", "pages": pages,
                           "chunks": retriever.vectorstore.index.ntotal,
                           "dimension": retriever.vectorstore.index.d, "fingerprint": fingerprint}

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    chunks = make_chunks(sources, document_dir, tokenizer, settings.chunk_tokens, settings.overlap_tokens)
    if not chunks:
        raise ValueError("PDF에서 색인할 텍스트 청크를 만들지 못했습니다")
    store = FAISS.from_documents(chunks, embedder)
    if store.index.ntotal != len(chunks):
        raise ValueError("색인 벡터 수와 청크 수가 다릅니다")

    index_dir.mkdir(parents=True, exist_ok=True)
    staged = index_dir / f"staged-{uuid.uuid4().hex}"
    try:
        store.save_local(str(staged))
        meta = {
            "format": INDEX_FORMAT, "created_by": "kv-cache-eval:technical-research",
            "technology": technology,
            "fingerprint": fingerprint, "model": settings.model_name,
            "chunk_tokens": settings.chunk_tokens, "overlap_tokens": settings.overlap_tokens,
            "pages": pages, "chunk_count": len(chunks), "dimension": store.index.d,
            "files": {name: file_sha256(staged / name) for name in INDEX_FILES},
        }
        (staged / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        for name in (*INDEX_FILES, "manifest.json"):
            os.replace(staged / name, index_dir / name)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return LocalRetriever(store), {"status": "built", "pages": pages,
                                  "chunks": len(chunks), "dimension": store.index.d,
                                  "fingerprint": fingerprint}
