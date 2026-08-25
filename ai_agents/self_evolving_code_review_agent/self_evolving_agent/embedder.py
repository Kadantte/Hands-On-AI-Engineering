"""Local BGE embeddings for code-review memory retrieval."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return int(self._model.get_embedding_dimension())
        return int(self._model.get_sentence_embedding_dimension())

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_document(self, text: str) -> list[float]:
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

