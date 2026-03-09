from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _get_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True)
class AgentConfig:
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    embedding_provider: str = "huggingface"
    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    collection_name: str = "nckh_database"
    similarity_top_k: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 50
    max_history_turns: int = 8
    max_context_chars: int = 7000
    max_sources_in_answer: int = 5
    rewrite_query: bool = True
    data_dir: Path = _project_root() / "data_raw"
    vector_store_dir: Path = _project_root() / "vector_store"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Missing GROQ_API_KEY. Set environment variable first, "
                "example PowerShell: $env:GROQ_API_KEY='your_key'"
            )
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
        temperature = _get_float_env("GROQ_TEMPERATURE", 0.1, 0.0, 1.0)
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
        embedding_model = os.getenv("EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder").strip()
        history_turns = _get_int_env("MAX_HISTORY_TURNS", 8, 0, 20)
        max_context_chars = _get_int_env("MAX_CONTEXT_CHARS", 7000, 1000, 20000)
        similarity_top_k = _get_int_env("SIMILARITY_TOP_K", 10, 1, 50)
        chunk_size = _get_int_env("CHUNK_SIZE", 1000, 200, 4000)
        chunk_overlap = _get_int_env("CHUNK_OVERLAP", 50, 0, 500)
        max_sources_in_answer = _get_int_env("MAX_SOURCES_IN_ANSWER", 5, 1, 10)
        rewrite_query = _get_bool_env("REWRITE_QUERY", True)
        collection_name = os.getenv("COLLECTION_NAME", "nckh_database").strip() or "nckh_database"
        data_dir = Path(os.getenv("DATA_DIR", str(_project_root() / "data_raw")).strip())
        vector_store_dir = Path(
            os.getenv("VECTOR_STORE_DIR", str(_project_root() / "vector_store")).strip()
        )
        return cls(
            groq_api_key=api_key,
            groq_model=model,
            temperature=temperature,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            collection_name=collection_name,
            similarity_top_k=similarity_top_k,
            chunk_size=chunk_size,
            chunk_overlap=min(chunk_overlap, max(0, chunk_size - 1)),
            max_history_turns=history_turns,
            max_context_chars=max_context_chars,
            max_sources_in_answer=max_sources_in_answer,
            rewrite_query=rewrite_query,
            data_dir=data_dir,
            vector_store_dir=vector_store_dir,
        )
