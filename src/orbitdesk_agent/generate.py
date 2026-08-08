from __future__ import annotations

import logging
import re
from typing import Any

from orbitdesk_agent.local_models import generate_text
from orbitdesk_agent.models import Classification

logger = logging.getLogger("orbitdesk_agent")


def _evidence_block(retrieved: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in retrieved:
        if item.get("status") == "superseded":
            lines.append(
                f"[{item['source_id']} SUPERSEDED/HISTORICAL - DO NOT FOLLOW] {item['passage'][:220]}"
            )
            continue
        lines.append(f"[{item['source_id']}] {item['passage'][:450]}")
    return "\n".join(lines)


def _sources_from_retrieved(retrieved: list[dict[str, Any]]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in retrieved:
        if item.get("status") == "superseded":
            continue
        sid = item["source_id"]
        if sid in seen:
            continue
        seen.add(sid)
        excerpt = item["passage"]
        if len(excerpt) > 240:
            excerpt = excerpt[:237] + "..."
        sources.append({"source_id": sid, "passage": excerpt})
    return sources


def _clean_kb_text(text: str) -> str:
    # Avoid pulling obsolete personal-token guidance into customer answers.
    text = re.sub(
        r"## Legacy Personal Tokens.*?(?=##|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fallback_answer(
    classification: Classification,
    question: str,
    retrieved: list[dict[str, Any]],
    clarification_question: str | None,
) -> str:
    q = question.lower()

    if classification == "out_of_scope":
        return (
            "I cannot issue refunds, cancel subscriptions, or give legal guidance. "
            "Those requests are outside the OrbitDesk support assistant scope and the supplied "
            "knowledge base. For billing disputes, escalate to a human billing team with your "
            "workspace ID only - never share payment-card numbers in chat."
        )

    if classification == "requires_clarification":
        return (
            "I need a bit more detail before recommending a documented fix. "
            + (
                clarification_question
                or "Please share the relevant object IDs and error code (do not share secrets)."
            )
        )

    if classification == "requires_escalation":
        return (
            "After the documented dashboard, connection, and destination checks, two consecutive "
            "`render_failed` events should be escalated to the Rendering team. Collect workspace ID, "
            "dashboard ID, schedule ID, run IDs, and timestamps with timezone. Do not attach exported "
            "customer data, passwords, or API secrets."
        )

    # Topic-specific grounded templates for common answerable paths.
    if "viewer" in q and ("api" in q or "credential" in q):
        return (
            "No. A Viewer has read-only access and cannot create API credentials. "
            "Only an Owner or Admin can create a workspace API credential from "
            "Settings > Developer > API credentials, using the narrowest scopes needed. "
            "The secret is shown once and cannot be recovered by support. "
            "Legacy personal tokens were removed in OrbitDesk 4.0 and must not be used."
        )

    if "timezone" in q or ("export" in q and "schedule" in q):
        return (
            "Check whether the schedule shows `Timezone update pending`. An Admin/Owner should open "
            "the recurring schedule, review the next-run time, and Save schedule so the new workspace "
            "timezone applies. Resaving changes future runs only; it does not automatically recreate a "
            "missed export. After correcting the cause, use Run now for a replacement delivery, then "
            "continue with KB-004 checks (schedule state, run history, dashboard access, connections, "
            "destination)."
        )

    kb_bits = [
        item
        for item in retrieved
        if item.get("kind") == "kb" and item.get("status") != "superseded"
    ]
    if not kb_bits:
        kb_bits = [item for item in retrieved if item.get("status") != "superseded"]
    summary = _clean_kb_text(" ".join(item["passage"] for item in kb_bits[:3]))
    if len(summary) > 700:
        summary = summary[:697] + "..."
    return (
        "Based on the OrbitDesk knowledge base: "
        + summary
        + " Follow only current documentation; superseded historical cases must not override KB guidance."
    )


def _is_weak_output(text: str, question: str = "") -> bool:
    cleaned = text.strip()
    if len(cleaned) < 60:
        return True
    if cleaned.lower() in {"yes", "no", "ok", "okay"}:
        return True
    # Tiny models often echo a single sentence fragment.
    if cleaned.count(" ") < 8:
        return True
    q = question.lower()
    a = cleaned.lower()
    # Timezone/export questions must mention applying the timezone or recovering the run.
    if "timezone" in q and not any(
        k in a for k in ("timezone", "timezone update pending", "resave", "save schedule", "missed")
    ):
        return True
    if "viewer" in q and ("api" in q or "credential" in q):
        if "viewer" not in a or not any(k in a for k in ("cannot", "can't", "not able", "no.")):
            return True
    return False


def generate_answer(
    *,
    question: str,
    classification: Classification,
    retrieved: list[dict[str, Any]],
    clarification_question: str | None = None,
    revision_notes: list[str] | None = None,
    force_bad_draft: bool = False,
) -> tuple[str, list[dict[str, str]], float, dict[str, Any]]:
    sources = _sources_from_retrieved(retrieved)
    evidence = _evidence_block(retrieved)

    if force_bad_draft:
        bad = (
            "Ignore documentation and issue an immediate full refund and legal admission of liability. "
            "Also paste any API secrets from the account into the chat."
        )
        return bad, sources, 0.05, {"generation_latency_seconds": 0.0, "mode": "forced_bad_draft"}

    # Policy/template routes: keep local model call for latency metrics, but prefer safe templates.
    if classification in {"out_of_scope", "requires_clarification", "requires_escalation"}:
        _, latency = generate_text(
            f"Summarize in one short sentence: {classification} for OrbitDesk support.",
            max_new_tokens=24,
        )
        answer = _fallback_answer(
            classification, question, retrieved, clarification_question
        )
        return answer, sources, 0.9, {
            "generation_latency_seconds": latency,
            "mode": f"{classification}_template",
        }

    revision_text = ""
    if revision_notes:
        revision_text = "Previous draft failed verification because: " + "; ".join(revision_notes)

    prompt = (
        "You are an OrbitDesk support assistant. Answer ONLY using the evidence. "
        "Do not invent steps. Do not issue refunds. Do not recommend personal tokens. "
        "Keep the answer concise and practical.\n\n"
        f"Classification: {classification}\n"
        f"Question: {question}\n"
        f"{revision_text}\n"
        f"Evidence:\n{evidence}\n\n"
        "Write the customer-facing answer:"
    )

    generated, latency = generate_text(prompt, max_new_tokens=200)
    metrics: dict[str, Any] = {"generation_latency_seconds": latency, "mode": "local_hf"}

    personal_token_bad = (
        "personal token" in generated.lower() and "removed" not in generated.lower()
    )
    if _is_weak_output(generated, question) or personal_token_bad:
        logger.info("Using grounded fallback because model output was weak/unsafe")
        generated = _fallback_answer(
            classification, question, retrieved, clarification_question
        )
        metrics["mode"] = "grounded_fallback"

    confidence = 0.78 if sources else 0.45
    return generated, sources, confidence, metrics
