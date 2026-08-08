from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
RESOLVED_CASES_PATH = DATA_DIR / "resolved_cases.json"
SAMPLE_QUESTIONS_PATH = DATA_DIR / "sample_questions.json"
OUTPUT_SCHEMA_PATH = DATA_DIR / "output_schema.json"
LOG_DIR = PROJECT_ROOT / "logs"

# Local Hugging Face models (pinned revisions for reproducibility)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
GENERATION_MODEL_NAME = "google/flan-t5-small"
GENERATION_MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"

TOP_K = 4
MAX_REVISION_ATTEMPTS = 1
MAX_GRAPH_STEPS = 12
