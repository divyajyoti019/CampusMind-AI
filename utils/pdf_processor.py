"""Page-aware PDF text extraction and overlapping chunk creation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf as fitz


DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into bounded character chunks with a fixed overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(normalized):
            break
        start += step
    return chunks


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract non-empty page text with 1-based page numbering and filename."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.name}")

    pages: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        "source": path.name,
                        "page": page_index,
                        "text": text,
                    }
                )
    return pages


def process_pdf(
    pdf_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Return text chunks with source, page, and stable local chunk IDs.

    Source/page aliases are included for compatibility with the broader
    application contract. Each chunk comes from exactly one PDF page so its
    citation remains page-specific.
    """
    chunks: list[dict[str, Any]] = []
    for page_data in extract_pdf_pages(pdf_path):
        page_chunks = split_text_into_chunks(
            page_data["text"], chunk_size=chunk_size, overlap=overlap
        )
        for chunk_number, text in enumerate(page_chunks, start=1):
            source = page_data["source"]
            page_number = page_data["page"]
            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "page": page_number,
                    "source_file": source,
                    "page_number": page_number,
                    "chunk_id": f"{source}:p{page_number}:c{chunk_number}",
                }
            )
    return chunks
