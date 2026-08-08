from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orbitdesk_agent.graph import run_agent
from orbitdesk_agent.local_models import load_models
from orbitdesk_agent.logging_utils import setup_logging


def _print_readable(result: dict) -> None:
    response = result.get("final_response") or {}
    print("\n=== Readable Answer ===")
    print(response.get("answer", ""))
    print("\n=== Structured JSON ===")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print("\n=== Execution Trace ===")
    print(" -> ".join(result.get("node_trace") or []))
    metrics = result.get("metrics") or {}
    if metrics:
        print("\n=== Metrics ===")
        print(json.dumps(metrics, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OrbitDesk local-first support agent (LangGraph + Hugging Face)"
    )
    parser.add_argument("question", nargs="?", help="Support question to answer")
    parser.add_argument(
        "-q",
        "--question-file",
        type=Path,
        help="Optional path to a text file containing the question",
    )
    parser.add_argument(
        "--force-verify-fail",
        action="store_true",
        help="Force the first generated draft to fail verification (demo/test path)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write full graph result JSON to this path",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load local models and exit (useful for offline demos)",
    )
    args = parser.parse_args(argv)

    logger = setup_logging()

    if args.preload:
        bundle = load_models()
        print(
            json.dumps(
                {
                    "embedding_model": f"{bundle.embedding_model_name}@{bundle.embedding_revision}",
                    "generation_model": f"{bundle.generation_model_name}@{bundle.generation_revision}",
                    "device": bundle.device,
                    "embed_load_seconds": bundle.embed_load_seconds,
                    "gen_load_seconds": bundle.gen_load_seconds,
                },
                indent=2,
            )
        )
        return 0

    question = args.question
    if args.question_file:
        question = args.question_file.read_text(encoding="utf-8").strip()
    if not question:
        question = input("Enter support question: ").strip()
    if not question:
        logger.error("No question provided.")
        return 2

    result = run_agent(question, force_verify_fail=args.force_verify_fail)
    _print_readable(result)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
