from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from orbitdesk_agent.config import KB_DIR, RESOLVED_CASES_PATH


@dataclass(frozen=True)
class Passage:
    source_id: str
    title: str
    filename: str
    text: str
    kind: str  # kb | case
    status: str
    tags: tuple[str, ...]


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_markdown(path: Path) -> Passage:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return Passage(
            source_id=path.stem,
            title=path.stem,
            filename=path.name,
            text=raw.strip(),
            kind="kb",
            status="current",
            tags=(),
        )

    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    tags = tuple(meta.get("tags") or [])
    return Passage(
        source_id=str(meta.get("document_id") or path.stem),
        title=str(meta.get("title") or path.stem),
        filename=path.name,
        text=body,
        kind="kb",
        status=str(meta.get("status") or "current"),
        tags=tags,
    )


def _chunk_text(text: str, max_chars: int = 700) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
            else:
                for i in range(0, len(paragraph), max_chars):
                    chunks.append(paragraph[i : i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def load_passages(
    kb_dir: Path | None = None,
    cases_path: Path | None = None,
) -> list[Passage]:
    kb_dir = kb_dir or KB_DIR
    cases_path = cases_path or RESOLVED_CASES_PATH
    passages: list[Passage] = []

    for path in sorted(kb_dir.glob("*.md")):
        doc = _parse_markdown(path)
        for idx, chunk in enumerate(_chunk_text(doc.text), start=1):
            passages.append(
                Passage(
                    source_id=doc.source_id,
                    title=doc.title,
                    filename=doc.filename,
                    text=chunk,
                    kind="kb",
                    status=doc.status,
                    tags=doc.tags,
                )
            )
            # keep chunk index in text prefix for traceability
            if idx > 1:
                passages[-1] = Passage(
                    source_id=doc.source_id,
                    title=f"{doc.title} (part {idx})",
                    filename=doc.filename,
                    text=chunk,
                    kind="kb",
                    status=doc.status,
                    tags=doc.tags,
                )

    payload: dict[str, Any] = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in payload.get("cases", []):
        status = str(case.get("status") or "resolved")
        body_parts = [
            f"Title: {case.get('title', '')}",
            "Symptoms: " + "; ".join(case.get("symptoms") or []),
            "Resolution: " + "; ".join(case.get("resolution") or []),
        ]
        if case.get("important_limit"):
            body_parts.append(f"Important limit: {case['important_limit']}")
        if case.get("superseded_reason"):
            body_parts.append(f"Superseded reason: {case['superseded_reason']}")
        text = "\n".join(body_parts)
        passages.append(
            Passage(
                source_id=str(case["case_id"]),
                title=str(case.get("title") or case["case_id"]),
                filename="resolved_cases.json",
                text=text,
                kind="case",
                status=status,
                tags=tuple(case.get("source_documents") or []),
            )
        )
    return passages
