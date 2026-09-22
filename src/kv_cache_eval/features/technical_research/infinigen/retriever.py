"""BGE-M3 dense 임베딩과 로컬 FAISS 색인. pickle 없이 검증 후 읽는다."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kv_cache_eval.common.config import get_embedding_model
from kv_cache_eval.features.technical_research.infinigen.ingest import (
    DEFAULT_DOCUMENT_DIR, Chunk, Source, file_sha256, load_sources, make_chunks,
    validate_sources,
)


DEFAULT_INDEX_DIR = Path("data/indexes/infinigen")
INDEX_FORMAT = 1


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


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def index_fingerprint(sources: list[Source], settings: RetrievalSettings) -> str:
    """문서 바이트, 모델, 청킹 정의가 바뀌면 값이 달라진다."""
    payload = {
        "format": INDEX_FORMAT,
        # 저장 청크의 인용 필드가 바뀌어도 이전 출처 메타데이터를 재사용하지 않는다.
        "sources": [
            (item.document_id, item.title, item.source_url, item.version,
             item.evidence_role, item.subject_technology, item.file,
             item.sha256, item.page_count)
            for item in sources
        ],
        "model": settings.model_name,
        "chunk_tokens": settings.chunk_tokens,
        "overlap_tokens": settings.overlap_tokens,
        "extraction": "pypdf-normalized-whitespace-fast-token-offset-v1",
        "vector": "dense-normalized-inner-product-v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def index_is_current(index_dir: Path, fingerprint: str) -> bool:
    """설정과 저장 파일 해시가 맞는 완결된 로컬 색인만 재사용한다."""
    try:
        meta = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        return (
            meta["format"] == INDEX_FORMAT and meta["fingerprint"] == fingerprint
            and all(file_sha256(index_dir / name) == meta["files"][name]
                    for name in ("vectors.faiss", "chunks.json"))
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


def _vector(values: list[float], expected_dimension: int | None = None) -> np.ndarray:
    import faiss

    vector = np.asarray(values, dtype="float32").reshape(1, -1)
    if not np.isfinite(vector).all() or vector.shape[1] == 0:
        raise ValueError("임베딩 벡터가 비어 있거나 유한하지 않습니다")
    if expected_dimension is not None and vector.shape[1] != expected_dimension:
        raise ValueError(f"임베딩 차원 불일치: {vector.shape[1]} != {expected_dimension}")
    faiss.normalize_L2(vector)
    return vector


class LocalRetriever:
    def __init__(self, index: Any, chunks: list[Chunk], embedder: Any):
        if index.ntotal != len(chunks):
            raise ValueError("FAISS 벡터 수와 청크 수가 다릅니다")
        self.index, self.chunks, self.embedder = index, chunks, embedder

    def search(self, query: str, top_k: int = 5, roles: set[str] | None = None) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k는 양수여야 합니다")
        vector = _vector(self.embedder.embed_query(query), self.index.d)
        # 소규모 원문 풀에서는 전체 후보를 점수화해 역할 필터를 정확히 적용한다.
        scores, positions = self.index.search(vector, self.index.ntotal)
        hits = [SearchHit(self.chunks[int(position)], float(score))
                for score, position in zip(scores[0], positions[0])
                if position >= 0 and (roles is None or self.chunks[int(position)].source_role in roles)]
        return hits[:top_k]


def _read_index(index_dir: Path, embedder: Any) -> LocalRetriever:
    import faiss

    meta = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
    index = faiss.read_index(str(index_dir / "vectors.faiss"))
    if index.d != meta["dimension"] or index.ntotal != meta["chunk_count"]:
        raise ValueError("저장된 색인의 차원 또는 청크 수가 manifest와 다릅니다")
    _vector(embedder.embed_query("dimension check"), index.d)
    return LocalRetriever(index, chunks, embedder)


def ensure_index(
    document_dir: Path = DEFAULT_DOCUMENT_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    settings: RetrievalSettings | None = None,
) -> tuple[LocalRetriever, dict[str, Any]]:
    """원문 검증 후 현재 색인을 재사용하거나 다시 만든다. 모델 다운로드는 여기서만 시작."""
    import faiss
    from transformers import AutoTokenizer

    settings = (settings or RetrievalSettings()).resolved()
    if settings.batch_size <= 0 or settings.chunk_tokens <= 0 or not 0 <= settings.overlap_tokens < settings.chunk_tokens:
        raise ValueError("검색/청킹 설정값이 유효하지 않습니다")
    if settings.device == "cpu":
        # 대형 임베딩 모델과 FAISS의 과도한 CPU 스레드 경합을 피한다.
        import torch
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
    sources, budget = load_sources()
    pages = validate_sources(sources, document_dir, budget)
    fingerprint = index_fingerprint(sources, settings)
    embedder = _embedder(settings)
    if index_is_current(index_dir, fingerprint):
        retriever = _read_index(index_dir, embedder)
        return retriever, {"status": "reused", "pages": pages, "chunks": len(retriever.chunks),
                           "dimension": retriever.index.d, "fingerprint": fingerprint}

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    chunks = make_chunks(sources, document_dir, tokenizer, settings.chunk_tokens, settings.overlap_tokens)
    if not chunks:
        raise ValueError("PDF에서 색인할 텍스트 청크를 만들지 못했습니다")
    index = None
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        vectors = np.asarray(embedder.embed_documents([item.text for item in batch]), dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(batch) or not np.isfinite(vectors).all():
            raise ValueError("문서 임베딩의 형태 또는 값이 유효하지 않습니다")
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[1] != index.d:
            raise ValueError("배치 간 임베딩 차원이 다릅니다")
        faiss.normalize_L2(vectors)
        index.add(vectors)
    assert index is not None
    index_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    staged_index = index_dir / f"vectors.{suffix}.tmp"
    staged_chunks = index_dir / f"chunks.{suffix}.tmp"
    staged_manifest = index_dir / f"manifest.{suffix}.tmp"
    try:
        faiss.write_index(index, str(staged_index))
        staged_chunks.write_text(json.dumps([item.to_dict() for item in chunks], ensure_ascii=False), encoding="utf-8")
        meta = {
            "format": INDEX_FORMAT, "fingerprint": fingerprint,
            "model": settings.model_name, "chunk_tokens": settings.chunk_tokens,
            "overlap_tokens": settings.overlap_tokens, "pages": pages,
            "chunk_count": len(chunks), "dimension": index.d,
            "files": {"vectors.faiss": file_sha256(staged_index), "chunks.json": file_sha256(staged_chunks)},
        }
        staged_manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(staged_index, index_dir / "vectors.faiss")
        os.replace(staged_chunks, index_dir / "chunks.json")
        os.replace(staged_manifest, index_dir / "manifest.json")
    finally:
        for path in (staged_index, staged_chunks, staged_manifest):
            path.unlink(missing_ok=True)
    return LocalRetriever(index, chunks, embedder), {
        "status": "built", "pages": pages, "chunks": len(chunks),
        "dimension": index.d, "fingerprint": fingerprint,
    }
