"""RAG grounding checks using deterministic embeddings and a mocked Gemini client."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from modules.rag_assistant import RAGAssistant
from prompts.rag_prompt import NO_ANSWER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = PROJECT_ROOT / "data" / "documents" / "University_Academic_Handbook.pdf"


class TinyEncoder:
    """Small word-bucket encoder to keep unit tests local and reproducible."""

    _terms = (
        "attendance", "75", "medical", "condonation", "exam", "grading", "fees",
        "deadline", "hostel", "library", "astronomy", "galaxy", "policy", "minimum",
    )

    def encode(self, sentences: list[str], **_: Any) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            vector = np.array(
                [1.0 if term in tokens else 0.0 for term in self._terms],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
            vectors.append(vector)
        return np.vstack(vectors)


class FakeGemini:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0
        self.models = self

    def generate_content(self, *, model: str, contents: str) -> Any:
        self.calls += 1
        assert model == "test-model"
        assert "75%" in contents
        return type("Response", (), {"text": self.response_text})()


@pytest.fixture
def indexed_assistant(tmp_path: Path) -> RAGAssistant:
    assistant = RAGAssistant(
        index_dir=tmp_path / "faiss_index",
        model_name="test-encoder",
        gemini_model="test-model",
        top_k=4,
        similarity_threshold=0.25,
        encoder=TinyEncoder(),
        client=FakeGemini(
            "Students must maintain at least 75% attendance in each course "
            "[University_Academic_Handbook.pdf, Page 1]."
        ),
    )
    assistant.ingest_documents(HANDBOOK)
    return assistant


def test_policy_answer_includes_expected_page_citation(indexed_assistant: RAGAssistant) -> None:
    answer = indexed_assistant.answer("What is the minimum attendance requirement?")
    assert "75%" in answer
    assert "[University_Academic_Handbook.pdf, Page 1]" in answer


def test_out_of_scope_question_is_refused_without_gemini_call(
    indexed_assistant: RAGAssistant,
) -> None:
    answer = indexed_assistant.answer("What is the nearest galaxy to the Milky Way?")
    assert answer == NO_ANSWER
    assert indexed_assistant.client.calls == 0
