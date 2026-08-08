from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from orbitdesk_agent.config import MAX_GRAPH_STEPS, MAX_REVISION_ATTEMPTS
from orbitdesk_agent.generate import generate_answer
from orbitdesk_agent.local_models import load_models
from orbitdesk_agent.models import AgentState
from orbitdesk_agent.retrieval import get_retriever
from orbitdesk_agent.triage import triage_question
from orbitdesk_agent.verify import build_response_dict, verify_response

logger = logging.getLogger("orbitdesk_agent")


def _log_node(name: str, state: AgentState) -> None:
    logger.info(
        "NODE=%s | classification=%s | revision=%s | question=%.80s",
        name,
        state.get("classification"),
        state.get("revision_count", 0),
        state.get("question", ""),
    )


def triage_node(state: AgentState) -> dict[str, Any]:
    _log_node("triage", state)
    result = triage_question(state["question"])
    return {
        "classification": result.classification,
        "reason": result.reason,
        "clarification_question": result.clarification_question,
        "requires_human": result.requires_human,
        "confidence": result.confidence,
        "revision_count": state.get("revision_count", 0),
        "warnings": [],
        "node_trace": ["triage"],
    }


def retrieve_node(state: AgentState) -> dict[str, Any]:
    _log_node("retrieve", state)
    retriever = get_retriever()
    hits = retriever.search(state["question"])
    retrieved = [
        {
            "source_id": h.source_id,
            "title": h.title,
            "filename": h.filename,
            "passage": h.passage,
            "kind": h.kind,
            "status": h.status,
            "score": h.score,
        }
        for h in hits
    ]
    logger.info(
        "Retrieved %d passages: %s",
        len(retrieved),
        ", ".join(f"{r['source_id']}({r['score']:.3f})" for r in retrieved),
    )
    warnings = list(state.get("warnings") or [])
    if any(r["status"] == "superseded" for r in retrieved):
        warnings.append("A superseded historical case was retrieved and must not override current KB.")
    return {
        "retrieved": retrieved,
        "warnings": warnings,
        "node_trace": ["retrieve"],
    }


def generate_node(state: AgentState) -> dict[str, Any]:
    _log_node("generate", state)
    force_bad = bool(state.get("force_verify_fail")) and state.get("revision_count", 0) == 0
    answer, sources, confidence, gen_metrics = generate_answer(
        question=state["question"],
        classification=state.get("classification", "answerable"),
        retrieved=state.get("retrieved") or [],
        clarification_question=state.get("clarification_question"),
        revision_notes=state.get("verification_notes"),
        force_bad_draft=force_bad,
    )
    metrics = dict(state.get("metrics") or {})
    metrics.update(gen_metrics)
    bundle = load_models()
    metrics.update(
        {
            "embedding_model": f"{bundle.embedding_model_name}@{bundle.embedding_revision}",
            "generation_model": f"{bundle.generation_model_name}@{bundle.generation_revision}",
            "device": bundle.device,
            "embed_load_seconds": bundle.embed_load_seconds,
            "gen_load_seconds": bundle.gen_load_seconds,
        }
    )
    # Keep triage confidence unless generation lowered it.
    final_conf = min(float(state.get("confidence") or 0.8), float(confidence))
    return {
        "draft_answer": answer,
        "sources": sources,
        "confidence": final_conf,
        "metrics": metrics,
        "node_trace": ["generate"],
    }


def verify_node(state: AgentState) -> dict[str, Any]:
    _log_node("verify", state)
    # On first pass with force flag, fail once; after revision, verify normally.
    force_fail = bool(state.get("force_verify_fail")) and state.get("revision_count", 0) == 0
    passed, notes = verify_response(state, force_fail=force_fail)
    logger.info("Verification passed=%s notes=%s", passed, notes)
    return {
        "verification_passed": passed,
        "verification_notes": notes,
        "node_trace": ["verify"],
    }


def revise_node(state: AgentState) -> dict[str, Any]:
    _log_node("revise", state)
    revision_count = int(state.get("revision_count") or 0) + 1
    warnings = list(state.get("warnings") or [])
    warnings.append("Initial draft failed verification; revising once.")
    return {
        "revision_count": revision_count,
        "warnings": warnings,
        "force_verify_fail": False,
        "node_trace": ["revise"],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    _log_node("finalize", state)
    payload = build_response_dict(state)
    if not state.get("verification_passed"):
        payload = {
            "classification": "safe_failure",
            "answer": (
                "I could not produce a fully verified answer from the available OrbitDesk evidence. "
                "Please rephrase with object IDs and error codes, or escalate to a human agent with "
                "workspace ID and the troubleshooting steps already tried. Do not share secrets."
            ),
            "sources": state.get("sources") or [],
            "confidence": 0.2,
            "requires_human": True,
            "reason": "Verification failed after revision/fallback; returning safe failure.",
            "clarification_question": state.get("clarification_question"),
            "warnings": (state.get("warnings") or []) + (state.get("verification_notes") or []),
        }
    else:
        # Ensure reason reflects route.
        payload["reason"] = state.get("reason") or payload["reason"]
    logger.info("FINAL classification=%s sources=%d", payload["classification"], len(payload["sources"]))
    return {
        "final_response": payload,
        "classification": payload["classification"],
        "draft_answer": payload["answer"],
        "confidence": payload["confidence"],
        "requires_human": payload["requires_human"],
        "reason": payload["reason"],
        "warnings": payload.get("warnings") or [],
        "node_trace": ["finalize"],
    }


def route_after_triage(
    state: AgentState,
) -> Literal["retrieve", "finalize_direct"]:
    classification = state.get("classification")
    if classification in {"out_of_scope", "requires_clarification"}:
        # Still retrieve for citations / evidence where useful, except pure policy refusals.
        if classification == "out_of_scope":
            return "retrieve"
        return "retrieve"
    return "retrieve"


def route_after_verify(
    state: AgentState,
) -> Literal["revise", "finalize"]:
    if state.get("verification_passed"):
        return "finalize"
    if int(state.get("revision_count") or 0) < MAX_REVISION_ATTEMPTS:
        return "revise"
    return "finalize"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "retrieve": "retrieve",
            "finalize_direct": "finalize",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "revise": "revise",
            "finalize": "finalize",
        },
    )
    graph.add_edge("revise", "generate")
    graph.add_edge("finalize", END)

    return graph.compile()


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run_agent(
    question: str,
    *,
    force_verify_fail: bool = False,
) -> dict[str, Any]:
    app = get_app()
    # Ensure models are warm before graph run for clearer latency logs.
    load_models()
    initial: AgentState = {
        "question": question,
        "revision_count": 0,
        "force_verify_fail": force_verify_fail,
        "node_trace": [],
        "warnings": [],
        "metrics": {},
    }
    # recursion_limit protects against infinite loops
    result = app.invoke(initial, config={"recursion_limit": MAX_GRAPH_STEPS})
    return result
