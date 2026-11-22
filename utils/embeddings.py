"""SentenceTransformer embedding and persisted FAISS index helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INDEX_NAME = "campusmind.faiss"
DEFAULT_METADATA_NAME = "chunks.json"


class TextEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> TextEncoder:
    """Load and cache a local SentenceTransformer model on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    *,
    model: TextEncoder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> np.ndarray:
    """Return normalized float32 vectors suitable for cosine/IP search."""
    if not texts:
        raise ValueError("At least one text is required to create embeddings")
    encoder = model or load_embedding_model(model_name)
    vectors = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _faiss_module() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on deployment setup
        raise RuntimeError(
            "faiss-cpu is required to build or load the CampusMind index"
        ) from exc
    return faiss


def build_faiss_index(
    chunks: list[dict[str, Any]],
    *,
    index_dir: str | Path,
    model: TextEncoder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    index_name: str = DEFAULT_INDEX_NAME,
    metadata_name: str = DEFAULT_METADATA_NAME,
) -> tuple[Any, list[dict[str, Any]]]:
    """Embed chunks, build an inner-product FAISS index, and persist both files."""
    if not chunks:
        raise ValueError("Cannot create a vector index without document chunks")
    if any(not isinstance(chunk.get("text"), str) or not chunk["text"].strip() for chunk in chunks):
        raise ValueError("Every indexed chunk must include non-empty text")

    vectors = embed_texts(
        [chunk["text"] for chunk in chunks], model=model, model_name=model_name
    )
    if vectors.ndim != 2:
        raise ValueError("The embedding model must return a two-dimensional matrix")

    faiss = _faiss_module()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    records = [dict(chunk) for chunk in chunks]
    destination = Path(index_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / index_name
    metadata_path = destination / metadata_name
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index, records


def load_faiss_index(
    index_dir: str | Path,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
    metadata_name: str = DEFAULT_METADATA_NAME,
) -> tuple[Any, list[dict[str, Any]]]:
    """Load a FAISS index and its aligned metadata records from disk."""
    directory = Path(index_dir).expanduser().resolve()
    index_path = directory / index_name
    metadata_path = directory / metadata_name
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"FAISS index is incomplete in {directory}; ingest PDF documents first"
        )

    faiss = _faiss_module()
    index = faiss.read_index(str(index_path))
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(records) != index.ntotal:
        raise ValueError(
            "FAISS index and chunk metadata have different record counts "
            f"({index.ntotal} vectors, {len(records)} records)"
        )
    return index, records
