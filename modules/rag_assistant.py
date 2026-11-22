"""RAG university assistant: ingest PDFs, retrieve evidence, cite Gemini answers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from config import settings
from prompts.rag_prompt import NO_ANSWER, build_rag_prompt
from utils.embeddings import (
    DEFAULT_MODEL_NAME,
    TextEncoder,
    build_faiss_index,
    embed_texts,
    load_faiss_index,
)
from utils.pdf_processor import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, process_pdf


_CITATION_PATTERN = re.compile(r"\[([^\],]+),\s*Page\s+(\d+)\]")


class RAGAssistant:
    """University policy RAG service backed by local embeddings and FAISS."""

    def __init__(
        self,
        *,
        index_dir: str | Path = settings.faiss_index_dir,
        model_name: str = DEFAULT_MODEL_NAME,
        gemini_model: str = settings.gemini_model,
        api_key: str | None = settings.gemini_api_key,
        top_k: int = settings.rag_top_k,
        similarity_threshold: float = settings.rag_similarity_threshold,
        encoder: TextEncoder | None = None,
        client: Any | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.model_name = model_name
        self.gemini_model = gemini_model
        self.api_key = api_key
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.encoder = encoder
        self.client = client
        self.index: Any | None = None
        self.records: list[dict[str, Any]] = []

    def ingest_documents(
        self,
        documents: str | Path | Sequence[str | Path],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        """Extract page-aware chunks from one PDF, many PDFs, or a directory."""
        if isinstance(documents, (str, Path)):
            path = Path(documents).expanduser().resolve()
            paths = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        else:
            paths = [Path(path).expanduser().resolve() for path in documents]
        if not paths:
            raise ValueError("No PDF documents were supplied for ingestion")

        chunks: list[dict[str, Any]] = []
        for path in paths:
            chunks.extend(
                process_pdf(path, chunk_size=chunk_size, overlap=overlap)
            )
        if not chunks:
            raise ValueError("The supplied PDFs contain no extractable text")

        self.index, self.records = build_faiss_index(
            chunks,
            index_dir=self.index_dir,
            model=self.encoder,
            model_name=self.model_name,
        )
        return len(chunks)

    def _ensure_index_loaded(self) -> None:
        if self.index is None:
            self.index, self.records = load_faiss_index(self.index_dir)

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the highest-cosine-similarity chunks above the score threshold."""
        if not question.strip():
            raise ValueError("question must not be empty")
        self._ensure_index_loaded()
        assert self.index is not None

        requested = self.top_k if top_k is None else top_k
        if requested < 1:
            raise ValueError("top_k must be positive")
        query_vector = embed_texts(
            [question], model=self.encoder, model_name=self.model_name
        )
        scores, indices = self.index.search(
            query_vector, min(requested, self.index.ntotal)
        )
        retrieved: list[dict[str, Any]] = []
        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0 or float(score) < self.similarity_threshold:
                continue
            record = dict(self.records[int(index_position)])
            record["score"] = float(score)
            retrieved.append(record)
        return retrieved

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Set it in the environment or .env."
            )
        from google import genai

        self.client = genai.Client(api_key=self.api_key)
        return self.client

    @staticmethod
    def _citations_are_grounded(
        answer: str, retrieved_chunks: list[dict[str, Any]]
    ) -> bool:
        expected = {
            (
                str(chunk.get("source") or chunk.get("source_file") or ""),
                str(chunk.get("page") or chunk.get("page_number") or ""),
            )
            for chunk in retrieved_chunks
        }
        citations = _CITATION_PATTERN.findall(answer)
        return bool(citations) and all(
            (source.strip(), page.strip()) in expected for source, page in citations
        )

    def answer_with_sources(
        self, question: str, *, role: str = "Student"
    ) -> dict[str, Any]:
        """Return the generated response together with its retrieved evidence."""
        chunks = self.retrieve(question)
        if not chunks:
            return {"answer": NO_ANSWER, "sources": []}

        prompt = build_rag_prompt(question, chunks, role=role)
        response = self._get_client().models.generate_content(
            model=self.gemini_model,
            contents=prompt,
        )
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            answer = NO_ANSWER
        elif answer != NO_ANSWER and not self._citations_are_grounded(answer, chunks):
            answer = NO_ANSWER
        return {"answer": answer, "sources": chunks}

    def answer(self, question: str, *, role: str = "Student") -> str:
        """Retrieve evidence and generate an answer with validated page citations."""
        return self.answer_with_sources(question, role=role)["answer"]


def ingest_sample_documents() -> int:
    """Convenience entry point to index PDFs in the configured documents folder."""
    assistant = RAGAssistant()
    return assistant.ingest_documents(settings.documents_dir)
