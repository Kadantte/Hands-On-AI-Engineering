"""Persistent ExpeL-inspired memory backed by Actian VectorAI DB."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from actian_vectorai import Distance, PointStruct, VectorAIClient, VectorParams

from self_evolving_agent.config import INSIGHTS_COLLECTION, TRAJECTORIES_COLLECTION
from self_evolving_agent.embedder import Embedder
from self_evolving_agent.models import CommentFeedback, LearnedInsight, RetrievedMemory, ReviewComment


class ReviewMemory:
    def __init__(
        self,
        client: VectorAIClient,
        embedder: Embedder,
        insight_top_k: int,
        trajectory_top_k: int,
        min_score: float,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.insight_top_k = insight_top_k
        self.trajectory_top_k = trajectory_top_k
        self.min_score = min_score

    def setup(self) -> None:
        vector_config = VectorParams(size=self.embedder.dimension, distance=Distance.Cosine)
        for collection in (INSIGHTS_COLLECTION, TRAJECTORIES_COLLECTION):
            if not self.client.collections.exists(collection):
                self.client.collections.create(collection, vectors_config=vector_config)

    def retrieve(self, diff_text: str) -> tuple[list[RetrievedMemory], list[RetrievedMemory]]:
        query_vector = self.embedder.embed_query(diff_text[:8000])
        insights = self._search(INSIGHTS_COLLECTION, query_vector, self.insight_top_k, "rule")
        trajectories = self._search(
            TRAJECTORIES_COLLECTION,
            query_vector,
            self.trajectory_top_k,
            "reflection_summary",
        )
        return insights, trajectories

    def _search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        text_key: str,
    ) -> list[RetrievedMemory]:
        if self.client.points.count(collection) == 0:
            return []
        hits = self.client.points.search(collection, vector=vector, limit=limit)
        return [
            RetrievedMemory(
                text=str((hit.payload or {}).get(text_key, "")),
                score=round(float(hit.score), 3),
                payload=hit.payload or {},
            )
            for hit in hits
            if float(hit.score) >= self.min_score
        ]

    def store_learning(
        self,
        review_id: str,
        diff_text: str,
        language: str,
        summary: str,
        comments: list[ReviewComment],
        feedback: list[CommentFeedback],
        insights: list[LearnedInsight],
    ) -> dict[str, int | float]:
        now = datetime.now(timezone.utc).isoformat()
        accepted = sum(item.action == "accept" for item in feedback)
        rejected = sum(item.action == "reject" for item in feedback)
        edited = sum(item.action == "edit" for item in feedback)
        total = len(feedback)

        trajectory_text = (
            f"Language: {language}\nReview summary: {summary}\n"
            f"Feedback: {accepted} accepted, {rejected} rejected, {edited} edited\n"
            f"Diff:\n{diff_text[:6000]}"
        )
        trajectory_payload = {
            "review_id": review_id,
            "language": language,
            "created_at": now,
            "review_summary": summary,
            "reflection_summary": summary,
            "diff_excerpt": diff_text[:4000],
            "comments_json": json.dumps([item.model_dump() for item in comments]),
            "feedback_json": json.dumps([item.model_dump() for item in feedback]),
            "accepted_count": accepted,
            "rejected_count": rejected,
            "edited_count": edited,
            "comment_count": total,
            "rejection_rate": round(rejected / total, 4) if total else 0.0,
        }
        self.client.points.upsert(
            TRAJECTORIES_COLLECTION,
            [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=self.embedder.embed_document(trajectory_text),
                    payload=trajectory_payload,
                )
            ],
        )

        insight_points = []
        for insight in insights:
            payload = {
                "insight_id": str(uuid.uuid4()),
                "rule": insight.rule,
                "rationale": insight.rationale,
                "polarity": insight.polarity,
                "scope": insight.scope,
                "confidence": insight.confidence,
                "source_review_id": review_id,
                "created_at": now,
            }
            insight_points.append(
                PointStruct(
                    id=payload["insight_id"],
                    vector=self.embedder.embed_document(
                        f"{insight.scope}: {insight.rule}\n{insight.rationale}"
                    ),
                    payload=payload,
                )
            )
        if insight_points:
            self.client.points.upsert(INSIGHTS_COLLECTION, insight_points)

        return {
            "accepted": accepted,
            "rejected": rejected,
            "edited": edited,
            "total": total,
            "rejection_rate": round(rejected / total, 4) if total else 0.0,
            "insights_added": len(insight_points),
        }

    def list_insights(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._scroll_payloads(INSIGHTS_COLLECTION, limit)

    def list_trajectories(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._scroll_payloads(TRAJECTORIES_COLLECTION, limit)
        return sorted(rows, key=lambda item: item.get("created_at", ""))

    def counts(self) -> dict[str, int]:
        return {
            "insights": self.client.points.count(INSIGHTS_COLLECTION),
            "trajectories": self.client.points.count(TRAJECTORIES_COLLECTION),
        }

    def reset(self) -> None:
        for collection in (INSIGHTS_COLLECTION, TRAJECTORIES_COLLECTION):
            if self.client.collections.exists(collection):
                self.client.collections.delete(collection)
        self.setup()

    def _scroll_payloads(self, collection: str, limit: int) -> list[dict[str, Any]]:
        if self.client.points.count(collection) == 0:
            return []

        rows: list[dict[str, Any]] = []
        offset = None
        while len(rows) < limit:
            page_size = min(100, limit - len(rows))
            points, next_offset = self.client.points.scroll(
                collection,
                offset=offset,
                limit=page_size,
                with_payload=True,
            )
            rows.extend(point.payload or {} for point in points)
            if next_offset is None or not points:
                break
            offset = next_offset
        return rows


def connect_client(url: str, access_token: str | None) -> VectorAIClient:
    kwargs: dict[str, str] = {"url": url}
    if access_token:
        kwargs["access_token"] = access_token
    client = VectorAIClient(**kwargs)
    client.connect()
    return client
