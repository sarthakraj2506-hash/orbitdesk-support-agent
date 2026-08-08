from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from orbitdesk_agent.models import Classification


OUT_OF_SCOPE_PATTERNS = [
    r"\brefund\b",
    r"\blegal advice\b",
    r"\blawsuit\b",
    r"\bmedical\b",
    r"ignore (the )?(supplied |provided )?documentation\b",
    r"\bcancel (my )?subscription\b",
    r"\bwrite legal\b",
]

ESCALATION_PATTERNS = [
    r"render_failed",
    r"two (export )?runs in a row",
    r"two consecutive",
    r"escalat",
    r"still failing after",
    r"connector_internal_error",
]

CLARIFICATION_PATTERNS = [
    r"\bsync is not working\b",
    r"\bdata sync\b",
    r"\bnot working\b",
    r"\bbroken\b",
    r"\bfix it\b",
]


@dataclass
class TriageResult:
    classification: Classification
    reason: str
    clarification_question: Optional[str]
    requires_human: bool
    confidence: float


def _match_any(text: str, patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def triage_question(question: str) -> TriageResult:
    q = question.strip()
    lowered = q.lower()

    oos = _match_any(lowered, OUT_OF_SCOPE_PATTERNS)
    if oos:
        return TriageResult(
            classification="out_of_scope",
            reason=(
                "Request asks for unsupported actions (refund/legal/ignore docs) "
                "outside OrbitDesk support scope."
            ),
            clarification_question=None,
            requires_human=True,
            confidence=0.95,
        )

    esc = _match_any(lowered, ESCALATION_PATTERNS)
    if esc or ("render_failed" in lowered and "checked" in lowered):
        return TriageResult(
            classification="requires_escalation",
            reason=(
                "Documented checks appear complete and the failure condition "
                "matches an escalation path."
            ),
            clarification_question=None,
            requires_human=True,
            confidence=0.9,
        )

    # Vague connection/sync questions without IDs/error codes need clarification.
    vague_sync = _match_any(lowered, CLARIFICATION_PATTERNS)
    has_ids = bool(
        re.search(
            r"(workspace id|connection id|error code|schedule id|dashboard id|run id)",
            lowered,
        )
    )
    if vague_sync and not has_ids and "timezone" not in lowered and "export" not in lowered:
        return TriageResult(
            classification="requires_clarification",
            reason="Question lacks connection identifiers and error details needed to diagnose.",
            clarification_question=(
                "Please share the workspace ID, connection name/ID, current connection state, "
                "last successful refresh time, and the latest error code (do not share secrets)."
            ),
            requires_human=False,
            confidence=0.88,
        )

    return TriageResult(
        classification="answerable",
        reason="Question maps to OrbitDesk product documentation topics.",
        clarification_question=None,
        requires_human=False,
        confidence=0.8,
    )
