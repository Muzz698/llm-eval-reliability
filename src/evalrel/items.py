"""Benchmark items. An item is a dict: id, subject, question, choices (list of 4), answer (0-3)."""
from __future__ import annotations

import json
import random
from pathlib import Path

LETTERS = "ABCD"


def sample_mmlu(per_subject: int = 10, seed: int = 0) -> list[dict]:
    """Stratified sample of the MMLU test set: `per_subject` items from each of the 57 subjects."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject: dict[str, list[int]] = {}
    for i, subject in enumerate(ds["subject"]):
        by_subject.setdefault(subject, []).append(i)
    rng = random.Random(seed)
    items = []
    for subject in sorted(by_subject):
        for i in sorted(rng.sample(by_subject[subject], per_subject)):
            row = ds[i]
            items.append({"id": f"{subject}_{i}", "subject": subject, "question": row["question"],
                          "choices": list(row["choices"]), "answer": int(row["answer"])})
    return items


def save_items(items: list[dict], path: str | Path) -> None:
    Path(path).write_text("\n".join(json.dumps(it) for it in items) + "\n")


def load_items(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def rotate(item: dict, k: int) -> tuple[list[str], int]:
    """Cyclically shift the options by k. Over k = 0..3 the correct answer visits every position
    exactly once, so position and content are fully counterbalanced."""
    n = len(item["choices"])
    choices = [item["choices"][(j - k) % n] for j in range(n)]
    return choices, (item["answer"] + k) % n
