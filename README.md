# OrbitDesk Support Agent Network

> **Note for Data Analyst / Business Analyst recruiters:** This is an **AI engineering side project** (LangGraph agents). For DA/BA hiring, please review `ecommerce-sales-analytics`, `cx-support-analytics`, `d2c-profitability-model`, and `sales-analytics-dashboard` first.

Local-first support agent for the **Tantrabodh AI / AI Engineer Internship** assignment.

The system answers OrbitDesk support questions using only the supplied knowledge base and resolved cases. After models are downloaded once, the workflow can run with network access disabled.

## What this builds

A **LangGraph** workflow with shared typed state, conditional routing, one revision/fallback path, execution logs, and a recursion limit to prevent infinite loops.

| Responsibility | Node(s) | Implementation |
|---|---|---|
| Triage | `triage` | Deterministic classifier → `answerable` / `requires_clarification` / `requires_escalation` / `out_of_scope` |
| Retrieval | `retrieve` | Local Hugging Face embeddings (`sentence-transformers`) over KB + cases |
| Response generation | `generate` | Local Hugging Face seq2seq model (`google/flan-t5-small`) + grounded fallback |
| Verification | `verify` → `revise` / `finalize` | JSON Schema + grounding/safety checks; revise once or `safe_failure` |

Graph diagram: [`docs/graph.png`](docs/graph.png)

## Models used

| Role | Model | Revision |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| Generation | `google/flan-t5-small` | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |

These are CPU-friendly. A CUDA GPU is used automatically when available.

## Hardware

- **Target profile:** Windows x64, Python 3.12+, 8+ GB RAM, CPU (GPU optional)
- **Machine used for this run:** MSI GF63 Thin 11UC, Intel i7-11800H, 8 GB RAM, NVIDIA RTX 3050 Laptop GPU (+ Intel UHD). Demo run used **CPU** for model inference.
- Exact load times and per-answer latency are printed in CLI metrics after each run

## Setup

```bash
cd orbitdesk-support-agent
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .

# Download / warm local models (needs network once)
python -m orbitdesk_agent.cli --preload
```

Assignment materials live under [`data/`](data/).

## Usage

Ask any natural-language question:

```bash
python -m orbitdesk_agent.cli "I am a read-only Viewer. Can I create an API credential?"
```

Demonstrate verification failure → revision:

```bash
python -m orbitdesk_agent.cli --force-verify-fail "Our daily dashboard exports stopped after a timezone change. What should we check?"
```

Run all five sample questions and write outputs:

```bash
python scripts/run_samples.py
```

Render the graph image:

```bash
python scripts/render_graph.py
```

## Output shape

Both a readable answer and structured JSON are returned:

```json
{
  "classification": "answerable",
  "answer": "...",
  "sources": [
    {"source_id": "KB-003", "passage": "..."}
  ],
  "confidence": 0.8,
  "requires_human": false,
  "reason": "...",
  "clarification_question": null,
  "warnings": []
}
```

Schema: [`data/output_schema.json`](data/output_schema.json)

## Required test-case coverage

| Case | Sample | Expected route |
|---|---|---|
| Directly answerable | Q-002 Viewer API credentials | `answerable` |
| Two documents | Q-001 timezone + exports | `answerable` (KB-003 + KB-004) |
| Clarification | Q-003 vague sync | `requires_clarification` |
| Out of scope | Q-005 refund/legal | `out_of_scope` |
| Verification failure | Q-001 with `--force-verify-fail` / sample runner | `verify` → `revise` → `generate` → `verify` → `finalize` |

Automated routing tests (no model wording dependency):

```bash
pytest -q
```

Sample run artifacts: [`samples/sample_outputs.json`](samples/sample_outputs.json)

## Design trade-offs

1. **Deterministic triage + local generation:** Triage uses explicit rules so routing is testable and stable; the HF model focuses on answer drafting from retrieved evidence.
2. **Grounded fallback:** Small CPU models can omit key steps, so generation falls back to evidence-based templates when output is too short/unsafe.
3. **No managed vector DB:** In-memory embeddings keep the demo fully local and simple.

## Known limitations

- `flan-t5-small` prose quality is limited; correctness of orchestration matters more here than polished writing.
- Retrieval is embedding similarity only (no cross-encoder reranker in the default path).
- The agent cannot inspect live OrbitDesk accounts; it only reasons over supplied documents.

## What I would improve with more time

- Add a local cross-encoder reranker
- Stronger claim-level entailment checks in verification
- Persist the embedding index to disk for faster cold starts
- Optional Gradio/Streamlit UI for the walkthrough video

## AI assistance disclosure

An AI coding assistant (Cursor) was used to scaffold the repository, implement the LangGraph workflow, write tests/docs, and iterate on local-model integration. Assignment materials and product rules come only from the provided package.

## Submission checklist

- [x] Source code + README + setup instructions
- [x] Test cases (`tests/test_routing.py`)
- [x] Sample outputs (`samples/`)
- [x] Graph diagram (`docs/graph.png`)
- [x] Exact model names + revisions (this README)
- [ ] GitHub repository link (push this project)
- [ ] 4–7 minute walkthrough video
- [ ] Hardware details for your machine in the Google Form
