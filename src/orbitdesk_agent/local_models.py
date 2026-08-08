from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from orbitdesk_agent.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    GENERATION_MODEL_NAME,
    GENERATION_MODEL_REVISION,
)

logger = logging.getLogger("orbitdesk_agent")


@dataclass
class ModelBundle:
    embedding_model_name: str
    embedding_revision: str
    generation_model_name: str
    generation_revision: str
    device: str
    embedder: SentenceTransformer
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    embed_load_seconds: float
    gen_load_seconds: float


_BUNDLE: ModelBundle | None = None


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(force_reload: bool = False) -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is not None and not force_reload:
        return _BUNDLE

    device = select_device()
    logger.info(
        "Loading embedding model %s@%s on %s",
        EMBEDDING_MODEL_NAME,
        EMBEDDING_MODEL_REVISION,
        device,
    )
    t0 = time.perf_counter()
    embedder = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        revision=EMBEDDING_MODEL_REVISION,
        device=device,
    )
    embed_load = time.perf_counter() - t0
    logger.info("Embedding model loaded in %.2fs", embed_load)

    logger.info(
        "Loading generation model %s@%s on %s",
        GENERATION_MODEL_NAME,
        GENERATION_MODEL_REVISION,
        device,
    )
    t1 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL_NAME,
        revision=GENERATION_MODEL_REVISION,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL_NAME,
        revision=GENERATION_MODEL_REVISION,
    )
    model.to(device)
    model.eval()
    gen_load = time.perf_counter() - t1
    logger.info("Generation model loaded in %.2fs", gen_load)

    _BUNDLE = ModelBundle(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        embedding_revision=EMBEDDING_MODEL_REVISION,
        generation_model_name=GENERATION_MODEL_NAME,
        generation_revision=GENERATION_MODEL_REVISION,
        device=device,
        embedder=embedder,
        tokenizer=tokenizer,
        model=model,
        embed_load_seconds=embed_load,
        gen_load_seconds=gen_load,
    )
    return _BUNDLE


def generate_text(prompt: str, max_new_tokens: int = 220) -> tuple[str, float]:
    bundle = load_models()
    t0 = time.perf_counter()
    encoded = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}
    with torch.no_grad():
        output_ids = bundle.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    latency = time.perf_counter() - t0
    return text, latency
