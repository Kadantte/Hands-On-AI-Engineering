"""Typed records exchanged by the graph, model, database, and UI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Severity = Literal["critical", "high", "medium", "low"]
FeedbackAction = Literal["accept", "reject", "edit"]


class ReviewComment(BaseModel):
    comment_id: str = ""
    line: str = Field(description="Changed line number or a short location label")
    severity: Severity
    category: str
    comment: str
    suggestion: str


class ReviewOutput(BaseModel):
    summary: str
    comments: list[ReviewComment]


class CommentFeedback(BaseModel):
    comment_id: str
    action: FeedbackAction
    edited_comment: str = ""
    note: str = ""


class LearnedInsight(BaseModel):
    rule: str
    rationale: str
    polarity: Literal["reinforce", "avoid", "refine"]
    scope: str = "general"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ReflectionOutput(BaseModel):
    summary: str
    insights: list[LearnedInsight]


class RetrievedMemory(BaseModel):
    text: str
    score: float
    payload: dict[str, Any]
