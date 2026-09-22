"""로컬 논문 PDF의 페이지 출처를 유지하는 최소 dense RAG 검색."""

import argparse
import json
from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kv_cache_eval.common.config import EMBEDDING_MODEL, MAX_SELECTED_SOURCE_PAGES
from kv_cache_eval.common.schemas import Technology

DOCUMENTS_DIR = Path(__file__).resolve().parents[4] / "data/documents"
DEFAULT_KIVI_PDF = DOCUMENTS_DIR / "kivi_original.pdf"
DEFAULT_VALIDATION_PDF = DOCUMENTS_DIR / "kvquant_validation.pdf"
SourceRole = Literal["primary", "independent_validation"]
LIMITATION_QUERY = "What are the limitations of KIVI?"
LIMITATION_SEARCH_QUERY = "KIVI 2bit Mistral may have a large accuracy drop 4bit needed"


def load_pdf(
    pdf_path: str | Path = DEFAULT_KIVI_PDF,
    *,
    technology: Technology = "KIVI",
    source_role: SourceRole | None = None,
) -> list[Document]:
    """PDF를 페이지별로 읽고 검색 결과에 필요한 출처 정보를 붙인다."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"논문 PDF를 찾을 수 없습니다: {path}")
    if source_role is None:
        source_role = "independent_validation" if path.name == DEFAULT_VALIDATION_PDF.name else "primary"

    pages = PyPDFLoader(str(path), mode="page").load()
    if len(pages) > MAX_SELECTED_SOURCE_PAGES:
        raise ValueError(f"선정 원문은 {MAX_SELECTED_SOURCE_PAGES}페이지 이하여야 합니다: {path}")

    for page in pages:
        page.metadata.update(
            technology=technology,
            source_id=path.stem,
            source_type="paper",
            source_role=source_role,
            source_technology="KVQuant" if source_role == "independent_validation" else technology,
            document=path.name,
            source=str(path),
            page=page.metadata["page"] + 1,  # PyPDFLoader의 0부터 시작하는 번호를 실제 페이지 번호로 변환
        )
    return pages


def build_index(
    pdf_path: str | Path = DEFAULT_KIVI_PDF,
    *,
    technology: Technology = "KIVI",
    include_validation: bool = False,
    validation_pdf_path: str | Path = DEFAULT_VALIDATION_PDF,
) -> FAISS:
    """BGE-M3 토큰 기준으로 분할한 뒤 FAISS dense 인덱스를 메모리에 만든다."""
    pages = load_pdf(pdf_path, technology=technology)
    if include_validation:
        pages.extend(load_pdf(validation_pdf_path, technology=technology, source_role="independent_validation"))
    if len(pages) > MAX_SELECTED_SOURCE_PAGES:
        raise ValueError(f"선정 원문은 총 {MAX_SELECTED_SOURCE_PAGES}페이지 이하여야 합니다")
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embeddings._client.tokenizer,
        chunk_size=512,
        chunk_overlap=64,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"PDF에서 검색 가능한 텍스트를 추출하지 못했습니다: {pdf_path}")
    return FAISS.from_documents(chunks, embeddings)


def retrieve(index: FAISS, query: str, *, k: int = 4, source_role: SourceRole = "primary") -> list[Document]:
    """문서 역할을 제한하고 원문 chunk 및 출처 metadata를 돌려준다."""
    if source_role == "primary" and query == LIMITATION_QUERY:
        # 원논문은 한계 절 대신 실험별 제약을 설명하므로 직접 보고한 조건으로 질의를 좁힌다.
        query = LIMITATION_SEARCH_QUERY
    return index.similarity_search(
        query, k=k, filter={"source_role": source_role}, fetch_k=index.index.ntotal
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="KIVI PDF의 BGE-M3 + FAISS 검색 확인")
    parser.add_argument("pdf_path", nargs="?", type=Path, default=DEFAULT_KIVI_PDF)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--validation-pdf", type=Path, default=DEFAULT_VALIDATION_PDF)
    args = parser.parse_args()

    index = build_index(
        args.pdf_path, include_validation=args.include_validation, validation_pdf_path=args.validation_pdf
    )
    queries = (
        ("How does KIVI quantize the key and value cache?", "primary"),
        ("What throughput improvement does KIVI report?", "primary"),
        (LIMITATION_QUERY, "primary"),
    )
    if args.include_validation:
        queries += (
            ("How was KIVI evaluated by independent follow-up work?", "independent_validation"),
            ("What limitations of KIVI were identified by KVQuant?", "independent_validation"),
        )
    for query, source_role in queries:
        search_query = query
        if source_role == "primary" and query == LIMITATION_QUERY:
            search_query = LIMITATION_SEARCH_QUERY
        results = [
            {
                "text": doc.page_content,
                "technology": doc.metadata["technology"],
                "source_id": doc.metadata["source_id"],
                "source_type": doc.metadata["source_type"],
                "source_role": doc.metadata["source_role"],
                "source_technology": doc.metadata["source_technology"],
                "document": doc.metadata["document"],
                "source": doc.metadata["source"],
                "page": doc.metadata["page"],
            }
            for doc in retrieve(index, query, source_role=source_role)
        ]
        print(json.dumps({"query": query, "search_query": search_query, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
