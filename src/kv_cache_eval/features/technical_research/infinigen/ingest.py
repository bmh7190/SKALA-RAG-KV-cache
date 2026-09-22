"""원문 검증과 물리 PDF 페이지 단위 청킹."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_id: str
    source_title: str
    source_url: str
    source_role: str
    subject_technology: str
    page: int  # 1-based physical PDF page, not printed paper page
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_chunks(
    sources: list[Source], document_dir: Path, tokenizer: Any, chunk_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    """실제 모델 tokenizer의 문자 offset을 이용해 페이지를 넘지 않는 청크를 만든다."""
    from pypdf import PdfReader

    if chunk_tokens <= 0 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("chunk_tokens > overlap_tokens >= 0 이어야 합니다")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("페이지 내 원문 offset 보존을 위해 fast tokenizer가 필요합니다")
    chunks: list[Chunk] = []
    step = chunk_tokens - overlap_tokens
    for source in sources:
        for page_number, page in enumerate(PdfReader(document_dir / source.file).pages, 1):
            text = _clean_text(page.extract_text() or "")
            if not text:
                continue
            offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for token_start in range(0, len(offsets), step):
                token_end = min(token_start + chunk_tokens, len(offsets))
                start = offsets[token_start][0]
                end = offsets[token_end - 1][1]
                piece = text[start:end].strip()
                if not piece:
                    continue
                stable = f"{source.document_id}:{page_number}:{start}:{end}:{piece}".encode("utf-8")
                chunks.append(Chunk(
                    id=hashlib.sha256(stable).hexdigest()[:20], text=piece,
                    source_id=source.document_id, source_title=source.title,
                    source_url=source.source_url, source_role=source.evidence_role,
                    subject_technology=source.subject_technology,
                    page=page_number, char_start=start, char_end=end,
                ))
                if token_end == len(offsets):
                    break
    return chunks
