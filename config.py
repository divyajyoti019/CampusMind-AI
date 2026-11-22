"""Central settings for CampusMind AI.

Environment variables override defaults. Keep secrets in the environment or in a
local, untracked .env file; never commit real API keys.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """Application paths, model identifiers, and safety/retrieval thresholds."""

    app_title: str = "CampusMind AI"
    app_icon: str = "🎓"

    # Gemini configuration. The key is optional at import time so the UI can
    # explain missing setup without preventing local/offline features from loading.
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    # Relative project data locations, resolved from this file rather than CWD.
    database_path: Path = PROJECT_ROOT / "data" / "campusmind.db"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    faiss_index_dir: Path = PROJECT_ROOT / "data" / "faiss_index"
    faiss_index_path: Path = PROJECT_ROOT / "data" / "faiss_index" / "campusmind.faiss"
    chunk_metadata_path: Path = PROJECT_ROOT / "data" / "faiss_index" / "chunks.json"

    # Local semantic retrieval configuration.
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.35
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Product policy and SQL safety defaults.
    sql_row_limit: int = 100
    low_attendance_threshold_pct: float = 75.0


settings = Settings()
