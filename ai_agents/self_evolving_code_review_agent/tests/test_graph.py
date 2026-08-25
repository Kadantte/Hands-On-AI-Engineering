"""Workflow tests that do not require Ollama or VectorAI DB."""

from __future__ import annotations

import unittest
from typing import Any

from self_evolving_agent.graph import build_graph, resume_review, start_review
from self_evolving_agent.models import (
    LearnedInsight,
    ReflectionOutput,
    ReviewComment,
    ReviewOutput,
)


class FakeMemory:
    """Small in-memory stand-in for the two persistent collections."""

    def __init__(self) -> None:
        self.saved: dict[str, Any] | None = None

    def retrieve(self, diff_text: str) -> tuple[list[Any], list[Any]]:
        return [], []

    def store_learning(self, **values: Any) -> dict[str, int | float]:
        self.saved = values
        feedback = values["feedback"]
        rejected = sum(item.action == "reject" for item in feedback)
        return {
            "insights_added": len(values["insights"]),
            "rejection_rate": rejected / len(feedback) if feedback else 0.0,
        }


class FakeLLM:
    """Return deterministic structured data for each agent stage."""

    def structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[Any],
        max_tokens: int = 2200,
    ) -> Any:
        if schema is ReviewOutput:
            return ReviewOutput(
                summary="One authorization issue was found.",
                comments=[
                    ReviewComment(
                        comment_id="comment-1",
                        line="delete_order",
                        severity="high",
                        category="security",
                        comment="The user identifier is not used before deletion.",
                        suggestion="Check ownership before deleting the order.",
                    )
                ],
            )
        if schema is ReflectionOutput:
            return ReflectionOutput(
                summary="Ownership checks are required for destructive operations.",
                insights=[
                    LearnedInsight(
                        rule="Require an ownership check before destructive database operations.",
                        polarity="reinforce",
                        scope="backend authorization",
                        rationale="The engineer accepted this security comment.",
                        confidence=0.9,
                    )
                ],
            )
        raise AssertionError(f"Unexpected schema: {schema}")


class ReviewGraphTests(unittest.TestCase):
    def test_review_pauses_for_feedback_then_persists_learning(self) -> None:
        memory = FakeMemory()
        graph = build_graph(memory, FakeLLM())  # type: ignore[arg-type]

        review_id, paused = start_review(
            graph,
            diff_text="def delete_order(order_id, user_id): pass",
            language="Python",
            team_context="Authorization is mandatory.",
        )

        self.assertEqual(len(paused["comments"]), 1)
        self.assertIn("__interrupt__", paused)

        completed = resume_review(
            graph,
            review_id,
            [
                {
                    "comment_id": "comment-1",
                    "action": "accept",
                    "edited_comment": "",
                    "note": "This matches our security standard.",
                }
            ],
        )

        self.assertEqual(completed["metrics"]["insights_added"], 1)
        self.assertIsNotNone(memory.saved)
        self.assertEqual(memory.saved["review_id"], review_id)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
