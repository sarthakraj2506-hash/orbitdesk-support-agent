from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbitdesk_agent.config import SAMPLE_QUESTIONS_PATH  # noqa: E402
from orbitdesk_agent.graph import run_agent  # noqa: E402
from orbitdesk_agent.logging_utils import setup_logging  # noqa: E402


def main() -> int:
    setup_logging()
    payload = json.loads(SAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))
    outputs = []
    for item in payload["questions"]:
        qid = item["question_id"]
        question = item["question"]
        force_fail = qid == "Q-001"  # demonstrate verify->revise path on first sample
        print(f"\n======== {qid} force_verify_fail={force_fail} ========")
        result = run_agent(question, force_verify_fail=force_fail)
        response = result.get("final_response") or {}
        print("TRACE:", " -> ".join(result.get("node_trace") or []))
        print("CLASS:", response.get("classification"))
        print(response.get("answer", "")[:500])
        outputs.append(
            {
                "question_id": qid,
                "question": question,
                "node_trace": result.get("node_trace"),
                "metrics": result.get("metrics"),
                "response": response,
            }
        )

    out_path = ROOT / "samples" / "sample_outputs.json"
    out_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
