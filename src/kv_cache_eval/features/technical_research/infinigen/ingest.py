"""원문 검증과 물리 PDF 페이지 단위 청킹."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document


SOURCES_FILE = Path(__file__).with_name("sources.json")
DEFAULT_DOCUMENT_DIR = Path("data/documents/infinigen")


@dataclass(frozen=True)
class Source:
    document_id: str
    title: str
    source_url: str
    version: str
    evidence_role: str
    subject_technology: str
    file: str
    page_count: int
    sha256: str


def load_sources(path: Path = SOURCES_FILE) -> tuple[list[Source], int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = [Source(**item) for item in manifest["documents"]]
    ids = [item.document_id for item in sources]
    if len(ids) != len(set(ids)) or sum(item.evidence_role == "primary" for item in sources) != 1:
        raise ValueError("출처 ID가 중복되거나 primary 문서가 하나가 아닙니다")
    if any(Path(item.file).name != item.file for item in sources):
        raise ValueError("출처 파일은 하위 경로 없이 파일명만 사용해야 합니다")
    return sources, int(manifest["page_budget"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sources(sources: list[Source], document_dir: Path, page_budget: int) -> int:
    """PDF 바이트 해시, 실제 페이지 수, 담당 예산을 검증한다."""
    from pypdf import PdfReader

    total = 0
    for source in sources:
        path = document_dir / source.file
        if not path.is_file():
            raise FileNotFoundError(f"원문 PDF가 없습니다: {path}")
        if file_sha256(path) != source.sha256:
            raise ValueError(f"원문 SHA256 불일치: {source.document_id}")
        count = len(PdfReader(path).pages)
        if count != source.page_count:
            raise ValueError(f"PDF 페이지 수 불일치: {source.document_id}: {count}")
        total += count
    if total > page_budget:
        raise ValueError(f"InfiniGen 원문 {total}쪽이 예산 {page_budget}쪽을 초과합니다")
    return total


def make_chunks(
    sources: list[Source], document_dir: Path, tokenizer: Any, chunk_tokens: int, overlap_tokens: int
) -> list[Document]:
    """LangChain PDF loader와 토큰 길이 splitter로 페이지별 청크를 만든다."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if chunk_tokens <= 0 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("chunk_tokens > overlap_tokens >= 0 이어야 합니다")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_tokens, chunk_overlap=overlap_tokens,
    )
    chunks: list[Document] = []
    for source in sources:
        pages = PyPDFLoader(str(document_dir / source.file), mode="page").load()
        if len(pages) != source.page_count:
            raise ValueError(f"PDF 로더 페이지 수 불일치: {source.document_id}")
        for page in pages:
            page_number = int(page.metadata["page"]) + 1
            for chunk_number, piece in enumerate(splitter.split_documents([page])):
                text = piece.page_content.strip()
                if not text:
                    continue
                stable = f"{source.document_id}:{page_number}:{chunk_number}:{text}".encode("utf-8")
                chunk_id = hashlib.sha256(stable).hexdigest()[:20]
                chunks.append(Document(id=chunk_id, page_content=text, metadata={
                    "chunk_id": chunk_id,
                    "source_id": source.document_id,
                    "source_title": source.title,
                    "source_url": source.source_url,
                    "source_role": source.evidence_role,
                    "subject_technology": source.subject_technology,
                    "page": page_number,  # 1-based physical PDF page
                }))
    return chunks
