from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from orbitdesk_agent.config import TOP_K
from orbitdesk_agent.corpus import Passage, load_passages
from orbitdesk_agent.local_models import load_models

logger = logging.getLogger("orbitdesk_agent")


@dataclass
class RetrievedPassage:
    source_id: str
    title: str
    filename: str
    passage: str
    kind: str
    status: str
    score: float


class LocalRetriever:
    def __init__(self) -> None:
        self.passages: list[Passage] = load_passages()
        self._matrix: np.ndarray | None = None

    def _ensure_index(self) -> np.ndarray:
        if self._matrix is not None:
            return self._matrix
        bundle = load_models()
        texts = [p.text for p in self.passages]
        logger.info("Embedding %d corpus passages", len(texts))
        matrix = bundle.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._matrix = np.asarray(matrix)
        return self._matrix

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedPassage]:
        matrix = self._ensure_index()
        bundle = load_models()
        query_vec = bundle.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = matrix @ query_vec
        ranked = np.argsort(-scores)

        results: list[RetrievedPassage] = []
        seen_keys: set[str] = set()
        for idx in ranked:
            passage = self.passages[int(idx)]
            # Prefer current KB over superseded cases when near-ties
            score = float(scores[int(idx)])
            if passage.kind == "case" and passage.status == "superseded":
                score -= 0.15
            key = f"{passage.source_id}:{passage.text[:80]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append(
                RetrievedPassage(
                    source_id=passage.source_id,
                    title=passage.title,
                    filename=passage.filename,
                    passage=passage.text,
                    kind=passage.kind,
                    status=passage.status,
                    score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results


_RETRIEVER: LocalRetriever | None = None


def get_retriever() -> LocalRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = LocalRetriever()
    return _RETRIEVER
