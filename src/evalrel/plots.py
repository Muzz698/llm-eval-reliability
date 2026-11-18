"""Figures for the reliability report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEMPLATE_MARKERS = {"plain": "o", "mmlu": "s", "paren": "^"}


def accuracy_by_condition(acc: pd.DataFrame, out: Path) -> None:
    """One row per model; each marker is one prompt template x option order."""
    models = sorted(acc["model"].unique(), key=lambda m: acc[acc.model == m].accuracy.mean())
    fig, ax = plt.subplots(figsize=(7, 0.7 + 0.55 * len(models)))
    rng = np.random.default_rng(0)
    for y, model in enumerate(models):
        g = acc[acc["model"] == model]
        for template, gt in g.groupby("template"):
            ax.scatter(gt["accuracy"] * 100, y + rng.uniform(-0.12, 0.12, len(gt)),
                       marker=TEMPLATE_MARKERS.get(template, "o"), s=28, alpha=0.75,
                       color="#4C72B0", label=template if y == 0 else None)
        ax.plot([g.accuracy.min() * 100, g.accuracy.max() * 100], [y, y], color="#999", lw=1, zorder=0)
    ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("MMLU accuracy (%) under one prompt template and option order")
    ax.legend(title="template", fontsize=8, title_fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def variance_shares(g: pd.DataFrame, out: Path) -> None:
    parts = [("i", "item", "#4C72B0"), ("t", "template", "#DD8452"), ("r", "option order", "#C44E52"),
             ("tr", "template x order", "#8172B3"), ("it", "item x template", "#55A868"),
             ("ir", "item x order", "#CCB974"), ("itr,e", "residual", "#BBBBBB")]
    fig, ax = plt.subplots(figsize=(7, 0.7 + 0.5 * len(g)))
    left = np.zeros(len(g))
    for key, label, color in parts:
        vals = g[f"pct_{key}"].to_numpy()
        ax.barh(g["model"], vals, left=left, color=color, label=label)
        left += vals
    ax.set_xlabel("share of variance in item correctness (%)")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def letter_choice(pb: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    width = 0.8 / len(pb)
    for k, (_, row) in enumerate(pb.iterrows()):
        ax.bar(np.arange(4) + k * width, [row[f"pick_{c}"] * 100 for c in "ABCD"], width,
               label=row["model"])
    ax.axhline(25, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(4) + 0.4 - width / 2, list("ABCD"))
    ax.set_ylabel("% of answers")
    ax.set_xlabel("chosen letter (correct answers are 25% each by design)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
