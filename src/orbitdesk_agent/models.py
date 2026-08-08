from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class SourceRef(BaseModel):
    source_id: str
    passage: str


class SupportResponse(BaseModel):
    classification: Classification
    answer: str = Field(min_length=1)
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human: bool
    reason: str = Field(min_length=1)
    clarification_question: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    question: str
    classification: Classification
    clarification_question: Optional[str]
    retrieved: list[dict[str, Any]]
    draft_answer: str
    sources: list[dict[str, str]]
    confidence: float
    requires_human: bool
    reason: str
    warnings: list[str]
    verification_passed: bool
    verification_notes: list[str]
    revision_count: int
    force_verify_fail: bool
    node_trace: Annotated[list[str], operator.add]
    metrics: dict[str, Any]
    final_response: dict[str, Any]
