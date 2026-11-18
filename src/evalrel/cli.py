"""Command line: evalrel sample | run | analyze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def cmd_sample(args) -> None:
    from .items import sample_mmlu, save_items

    items = sample_mmlu(per_subject=args.per_subject, seed=args.seed)
    save_items(items, args.out)
    print(f"Saved {len(items)} items to {args.out}")


def cmd_run(args) -> None:
    from .items import load_items
    from .llm import OllamaClient
    from .runner import run

    cfg = _cfg(args.config)
    items = load_items(cfg["items"])
    if args.n_items:
        items = items[:: max(1, len(items) // args.n_items)][: args.n_items]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models or cfg["models"]:
        out = out_dir / f"{model.replace(':', '-').replace('/', '_')}.csv"
        print(f"{model}: {len(items)} items -> {out}", flush=True)
        run(OllamaClient(model), items, cfg, out)


def cmd_analyze(args) -> None:
    import pandas as pd

    from . import metrics as M
    from . import plots

    cfg = _cfg(args.config)
    paths = sorted(Path(args.results).glob("*.csv"))
    df = M.load_results(paths)
    out = Path(args.results)
    tables = {
        "condition_tests": M.condition_tests(df),
        "spread": M.accuracy_spread(df), "g_study": M.g_study(df, cfg["benchmark_size"]),
        "consistency": M.item_consistency(df), "position_bias": M.position_bias(df),
        "retest": M.retest(df), "parse_rates": M.parse_rates(df),
    }
    if df["model"].nunique() > 1:
        tables["rank_flips"] = M.rank_flips(df)
    pd.set_option("display.width", 160, "display.max_columns", 30, "display.precision", 3)
    for name, t in tables.items():
        print(f"\n== {name} ==\n{t.to_string(index=False)}")
    (out / "analysis.json").write_text(json.dumps(
        {k: v.to_dict(orient="records") for k, v in tables.items()}, indent=1, default=float))
    plots.accuracy_by_condition(M.condition_accuracy(df), out / "accuracy_by_condition.png")
    plots.variance_shares(tables["g_study"], out / "variance_shares.png")
    plots.letter_choice(tables["position_bias"], out / "letter_choice.png")
    print(f"\nSaved analysis.json and figures to {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="evalrel", description=__doc__)
    p.add_argument("--config", default="configs/design.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the stratified MMLU item sample")
    s.add_argument("--per-subject", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/mmlu_570.jsonl")
    s.set_defaults(func=cmd_sample)
    r = sub.add_parser("run", help="query models over all conditions (resumable)")
    r.add_argument("--models", nargs="*")
    r.add_argument("--n-items", type=int, help="spread-out subset, for a pilot")
    r.add_argument("--out", default="results/main")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("analyze", help="reliability tables and figures")
    a.add_argument("results", nargs="?", default="results/main")
    a.set_defaults(func=cmd_analyze)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
