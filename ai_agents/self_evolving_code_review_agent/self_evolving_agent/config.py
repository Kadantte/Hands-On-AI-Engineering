"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    actian_url: str
    actian_access_token: str | None
    ollama_host: str
    ollama_model: str
    embedding_model: str
    insight_top_k: int
    trajectory_top_k: int
    min_relevance_score: float


def load_settings() -> Settings:
    """Load settings after dotenv has populated the process environment."""
    return Settings(
        actian_url=_required("ACTIAN_VECTORAI_URL"),
        actian_access_token=os.getenv("ACTIAN_VECTORAI_ACCESS_TOKEN") or None,
        ollama_host=_required("OLLAMA_HOST"),
        ollama_model=_required("OLLAMA_MODEL"),
        embedding_model=_required("EMBEDDING_MODEL"),
        insight_top_k=int(os.getenv("INSIGHT_TOP_K", "6")),
        trajectory_top_k=int(os.getenv("TRAJECTORY_TOP_K", "3")),
        min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.30")),
    )


INSIGHTS_COLLECTION = "review_insights"
TRAJECTORIES_COLLECTION = "review_trajectories"

