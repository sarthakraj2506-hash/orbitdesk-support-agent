from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, text, color="#E8F1F8"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        1.8,
        0.7,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#234E70",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + 0.9, y + 0.35, text, ha="center", va="center", fontsize=10, color="#102A43")


def arrow(ax, start, end, label=""):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#334E68",
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.12, label, fontsize=8, color="#486581", ha="center")


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("OrbitDesk Support Agent Graph", fontsize=14, pad=12)

    box(ax, (4.1, 6.0), "START", "#F0F4F8")
    box(ax, (4.1, 4.9), "triage\n(classify)", "#D9E2EC")
    box(ax, (4.1, 3.8), "retrieve\n(HF embeddings)", "#BCCCDC")
    box(ax, (4.1, 2.7), "generate\n(local Flan-T5)", "#9FB3C8")
    box(ax, (4.1, 1.6), "verify\n(schema+grounding)", "#829AB1")
    box(ax, (1.2, 1.6), "revise\n(once)", "#F8E3C5")
    box(ax, (7.0, 1.6), "finalize /\nsafe_failure", "#D9E8D5")
    box(ax, (7.0, 0.3), "END", "#F0F4F8")

    arrow(ax, (5.0, 6.0), (5.0, 5.6))
    arrow(ax, (5.0, 4.9), (5.0, 4.5))
    arrow(ax, (5.0, 3.8), (5.0, 3.4))
    arrow(ax, (5.0, 2.7), (5.0, 2.3))
    arrow(ax, (4.1, 1.95), (3.0, 1.95), "fail & retries left")
    arrow(ax, (2.1, 2.3), (4.2, 3.0), "retry generate")
    arrow(ax, (5.9, 1.95), (7.0, 1.95), "pass or no retries")
    arrow(ax, (7.9, 1.6), (7.9, 1.0))

    ax.text(
        0.4,
        0.35,
        "Shared typed state · Conditional routing · One revision fallback · recursion_limit guard",
        fontsize=9,
        color="#243B53",
    )

    out = Path(__file__).resolve().parents[1] / "docs" / "graph.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
