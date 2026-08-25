"""LangGraph retrieve, review, human-feedback, reflect, and persist workflow."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from self_evolving_agent.models import (
    CommentFeedback,
    LearnedInsight,
    ReflectionOutput,
    ReviewComment,
    ReviewOutput,
)

if TYPE_CHECKING:
    from self_evolving_agent.llm import LocalLLM
    from self_evolving_agent.memory import ReviewMemory
from self_evolving_agent.prompts import REFLECTION_SYSTEM, REVIEW_SYSTEM


class ReviewState(TypedDict, total=False):
    review_id: str
    diff_text: str
    language: str
    team_context: str
    insights: list[dict]
    trajectories: list[dict]
    summary: str
    comments: list[dict]
    feedback: list[dict]
    reflection_summary: str
    new_insights: list[dict]
    metrics: dict


def _feedback_node(state: ReviewState) -> dict[str, Any]:
    feedback = interrupt(
        {
            "review_id": state["review_id"],
            "instruction": "Accept, reject, or edit every review comment.",
            "comments": state.get("comments", []),
        }
    )
    return {"feedback": feedback}


def build_graph(memory: ReviewMemory, llm: LocalLLM) -> CompiledStateGraph:
    def retrieve_node(state: ReviewState) -> dict[str, Any]:
        insights, trajectories = memory.retrieve(state["diff_text"])
        return {
            "insights": [item.model_dump() for item in insights],
            "trajectories": [item.model_dump() for item in trajectories],
        }

    def review_node(state: ReviewState) -> dict[str, Any]:
        insight_text = "\n".join(
            f"- [{item['payload'].get('polarity', 'reinforce')}] {item['text']}"
            for item in state.get("insights", [])
        ) or "No learned insights yet."
        trajectory_text = "\n".join(
            f"- {item['text']}" for item in state.get("trajectories", [])
        ) or "No similar past reviews yet."

        output = llm.structured(
            REVIEW_SYSTEM,
            (
                f"Language: {state['language']}\n"
                f"Team context: {state.get('team_context') or 'Not provided'}\n\n"
                f"Learned insights:\n{insight_text}\n\n"
                f"Similar past reviews:\n{trajectory_text}\n\n"
                f"Diff or code:\n{state['diff_text']}"
            ),
            ReviewOutput,
        )
        comments = []
        for item in output.comments:
            comment = item.model_copy(
                update={"comment_id": item.comment_id or uuid.uuid4().hex[:10]}
            )
            comments.append(comment.model_dump())
        return {"summary": output.summary, "comments": comments}

    def reflect_node(state: ReviewState) -> dict[str, Any]:
        comments = [ReviewComment.model_validate(item) for item in state.get("comments", [])]
        feedback = [CommentFeedback.model_validate(item) for item in state.get("feedback", [])]
        if not comments:
            return {
                "reflection_summary": "No actionable comments were generated, so no new team rule was distilled.",
                "new_insights": [],
            }
        comment_map = {item.comment_id: item for item in comments}
        paired = []
        for item in feedback:
            comment = comment_map[item.comment_id]
            paired.append(
                {
                    "generated_comment": comment.model_dump(),
                    "engineer_action": item.action,
                    "engineer_edit": item.edited_comment,
                    "engineer_note": item.note,
                }
            )

        output = llm.structured(
            REFLECTION_SYSTEM,
            (
                f"Language: {state['language']}\n"
                f"Team context: {state.get('team_context') or 'Not provided'}\n"
                f"Review feedback:\n{json.dumps(paired, indent=2)}"
            ),
            ReflectionOutput,
            max_tokens=1400,
        )
        return {
            "reflection_summary": output.summary,
            "new_insights": [item.model_dump() for item in output.insights],
        }

    def persist_node(state: ReviewState) -> dict[str, Any]:
        comments = [ReviewComment.model_validate(item) for item in state.get("comments", [])]
        feedback = [CommentFeedback.model_validate(item) for item in state.get("feedback", [])]
        insights = [LearnedInsight.model_validate(item) for item in state.get("new_insights", [])]
        metrics = memory.store_learning(
            review_id=state["review_id"],
            diff_text=state["diff_text"],
            language=state["language"],
            summary=state.get("reflection_summary", ""),
            comments=comments,
            feedback=feedback,
            insights=insights,
        )
        return {"metrics": metrics}

    builder = StateGraph(ReviewState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("review", review_node)
    builder.add_node("human_feedback", _feedback_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("persist", persist_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "review")
    builder.add_edge("review", "human_feedback")
    builder.add_edge("human_feedback", "reflect")
    builder.add_edge("reflect", "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=MemorySaver())


def start_review(
    graph: CompiledStateGraph,
    diff_text: str,
    language: str,
    team_context: str,
) -> tuple[str, dict[str, Any]]:
    review_id = uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": review_id}}
    result = graph.invoke(
        {
            "review_id": review_id,
            "diff_text": diff_text,
            "language": language,
            "team_context": team_context,
        },
        config=config,
    )
    return review_id, result


def resume_review(
    graph: CompiledStateGraph,
    review_id: str,
    feedback: list[dict[str, Any]],
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": review_id}}
    return graph.invoke(Command(resume=feedback), config=config)
