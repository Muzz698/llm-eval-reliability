"""Run a model over items x conditions and append one row per response to a CSV.
Runs are resumable: rows already in the file are skipped."""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .items import LETTERS, rotate
from .prompts import TEMPLATES, parse_letter

FIELDS = ["model", "arm", "item", "subject", "template", "rotation", "temperature", "seed",
          "correct_letter", "response", "choice", "parsed", "correct", "orig_choice"]


def conditions(design: dict) -> list[dict]:
    """The full list of conditions: the greedy facet design plus the test-retest arm."""
    conds = []
    for template in design["templates"]:
        for rotation in design["rotations"]:
            conds.append({"arm": "facets", "template": template, "rotation": rotation,
                          "temperature": 0.0, "seed": 0})
    rt = design["retest"]
    for seed in rt["seeds"]:
        conds.append({"arm": "retest", "template": rt["template"], "rotation": rt["rotation"],
                      "temperature": rt["temperature"], "seed": seed})
    return conds


def _key(row: dict) -> tuple:
    return (row["model"], row["arm"], row["item"], row["template"], str(row["rotation"]),
            str(row["seed"]))


def run(client, items: list[dict], design: dict, out: str | Path, log_every: int = 200) -> Path:
    out = Path(out)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {_key(r) for r in csv.DictReader(f)}
    new_file = not out.exists()
    conds = conditions(design)
    total, n, t0 = len(items) * len(conds), 0, time.time()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for c in conds:
            for item in items:
                row = {"model": client.model, "arm": c["arm"], "item": item["id"],
                       "template": c["template"], "rotation": c["rotation"], "seed": c["seed"]}
                n += 1
                if _key({k: str(v) for k, v in row.items()}) in done:
                    continue
                choices, answer = rotate(item, c["rotation"])
                prompt = TEMPLATES[c["template"]](item, choices)
                response = client.generate(prompt, temperature=c["temperature"], seed=c["seed"])
                letter = parse_letter(response)
                pos = LETTERS.index(letter) if letter else None
                writer.writerow({
                    **row, "subject": item["subject"], "temperature": c["temperature"],
                    "correct_letter": LETTERS[answer], "response": response.strip()[:80],
                    "choice": letter or "", "parsed": int(letter is not None),
                    "correct": int(pos == answer),
                    # which ORIGINAL option was chosen, to compare content across rotations
                    "orig_choice": "" if pos is None else (pos - c["rotation"]) % 4,
                })
                f.flush()
                if n % log_every == 0:
                    rate = n / (time.time() - t0)
                    print(f"  {client.model}: {n}/{total} ({rate:.1f}/s)", flush=True)
    return out
