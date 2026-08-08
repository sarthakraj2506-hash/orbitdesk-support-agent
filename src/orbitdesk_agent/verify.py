from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema

from orbitdesk_agent.config import OUTPUT_SCHEMA_PATH
from orbitdesk_agent.models import SupportResponse


FORBIDDEN_CLAIM_PATTERNS = [
    r"\bi (have )?issued a refund\b",
    r"\bhere is (your )?api (secret|token|key)\b",
    r"\bpaste (your )?password\b",
    r"\bhere is (my |the )?legal advice\b",
    r"\byou must (get|receive) a refund\b",
    r"Profile > Personal token",
]


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{4,}", text.lower()) if t not in {"that", "this", "with", "from", "your", "have"}}


def build_response_dict(state: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "classification": state.get("classification", "safe_failure"),
        "answer": state.get("draft_answer") or "Unable to produce a grounded answer.",
        "sources": state.get("sources") or [],
        "confidence": float(state.get("confidence") or 0.0),
        "requires_human": bool(state.get("requires_human")),
        "reason": state.get("reason") or "No reason provided.",
        "clarification_question": state.get("clarification_question"),
        "warnings": state.get("warnings") or [],
    }
    return payload


def validate_schema(payload: dict[str, Any], schema_path: Path | None = None) -> list[str]:
    schema = json.loads((schema_path or OUTPUT_SCHEMA_PATH).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'.'.join(map(str, e.path)) or 'root'}: {e.message}" for e in validator.iter_errors(payload)]


def verify_response(state: dict[str, Any], force_fail: bool = False) -> tuple[bool, list[str]]:
    notes: list[str] = []
    payload = build_response_dict(state)

    if force_fail or state.get("force_verify_fail"):
        notes.append("Forced verification failure for demo/test path.")
        return False, notes

    schema_errors = validate_schema(payload)
    if schema_errors:
        notes.extend(schema_errors)
        return False, notes

    try:
        SupportResponse.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Pydantic validation failed: {exc}")
        return False, notes

    answer = payload["answer"]
    classification = payload["classification"]
    sources = payload["sources"]

    if not sources and classification in {"answerable", "requires_escalation"}:
        notes.append("Missing source references for an evidence-backed classification.")

    for pattern in FORBIDDEN_CLAIM_PATTERNS:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            notes.append(f"Unsafe or unsupported claim matched pattern: {pattern}")

    if "personal token" in answer.lower() and "removed" not in answer.lower() and "obsolete" not in answer.lower():
        notes.append("Answer references obsolete personal tokens as if still valid.")

    evidence_text = " ".join(item.get("passage", "") for item in state.get("retrieved") or [])
    evidence_text += " " + " ".join(s.get("passage", "") for s in sources)
    if classification == "answerable" and evidence_text.strip():
        overlap = len(_token_set(answer) & _token_set(evidence_text))
        if overlap < 3 and len(answer) > 80:
            notes.append("Answer appears weakly grounded in retrieved evidence.")

    if classification == "requires_clarification" and not payload.get("clarification_question"):
        notes.append("Clarification route missing clarification_question.")

    if classification == "out_of_scope" and "outside" not in answer.lower() and "cannot" not in answer.lower():
        notes.append("Out-of-scope answer does not clearly refuse unsupported action.")

    return (len(notes) == 0), notes
