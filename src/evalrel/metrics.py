"""Psychometric summaries of benchmark responses.

Input is the long table written by runner.run (one row per model x item x condition).
All functions are pure and are tested on synthetic data in tests/.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

Z95 = 1.959964


def load_results(paths) -> pd.DataFrame:
    """Read runner CSVs. orig_choice becomes a string '0'-'3', or '' when unparsed."""
    df = pd.concat([pd.read_csv(p, dtype={"choice": str}, keep_default_na=False) for p in paths],
                   ignore_index=True)
    df["orig_choice"] = df["orig_choice"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in ("rotation", "seed", "parsed", "correct"):
        df[col] = df[col].astype(int)
    return df


# ---------------------------------------------------------------- condition-level accuracy
def condition_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of each model under each (template, rotation) condition of the facet arm."""
    f = df[df["arm"] == "facets"]
    return (f.groupby(["model", "template", "rotation"])["correct"].mean()
            .rename("accuracy").reset_index())


def accuracy_spread(df: pd.DataFrame) -> pd.DataFrame:
    acc = condition_accuracy(df)
    n_items = df[df["arm"] == "facets"].groupby("model")["item"].nunique()
    out = acc.groupby("model")["accuracy"].agg(["mean", "std", "min", "max"])
    out["range"] = out["max"] - out["min"]
    out["n_conditions"] = acc.groupby("model").size()
    out["n_items"] = n_items
    # binomial standard error of ONE condition's accuracy, for comparison with the format SD
    out["item_se"] = np.sqrt(out["mean"] * (1 - out["mean"]) / out["n_items"])
    return out.reset_index()


# ---------------------------------------------------------------- generalizability theory
def variance_components(y: np.ndarray) -> dict[str, float]:
    """Random-effects G-study for a fully crossed items x templates x rotations design with one
    observation per cell (Brennan, 2001). y has shape (n_i, n_t, n_r). Negative estimates are set
    to zero, as is conventional. The three-way interaction is confounded with error."""
    n_i, n_t, n_r = y.shape
    g = y.mean()
    m_i, m_t, m_r = y.mean((1, 2)), y.mean((0, 2)), y.mean((0, 1))
    m_it, m_ir, m_tr = y.mean(2), y.mean(1), y.mean(0)

    ss_i = n_t * n_r * ((m_i - g) ** 2).sum()
    ss_t = n_i * n_r * ((m_t - g) ** 2).sum()
    ss_r = n_i * n_t * ((m_r - g) ** 2).sum()
    ss_it = n_r * ((m_it - m_i[:, None] - m_t[None, :] + g) ** 2).sum()
    ss_ir = n_t * ((m_ir - m_i[:, None] - m_r[None, :] + g) ** 2).sum()
    ss_tr = n_i * ((m_tr - m_t[:, None] - m_r[None, :] + g) ** 2).sum()
    resid = (y - m_it[:, :, None] - m_ir[:, None, :] - m_tr[None, :, :]
             + m_i[:, None, None] + m_t[None, :, None] + m_r[None, None, :] - g)
    ss_e = (resid ** 2).sum()

    def ms(ss, df):
        return ss / df if df > 0 else 0.0

    MS = {
        "i": ms(ss_i, n_i - 1), "t": ms(ss_t, n_t - 1), "r": ms(ss_r, n_r - 1),
        "it": ms(ss_it, (n_i - 1) * (n_t - 1)), "ir": ms(ss_ir, (n_i - 1) * (n_r - 1)),
        "tr": ms(ss_tr, (n_t - 1) * (n_r - 1)),
        "itr,e": ms(ss_e, (n_i - 1) * (n_t - 1) * (n_r - 1)),
    }
    v = {"itr,e": MS["itr,e"]}
    v["it"] = (MS["it"] - MS["itr,e"]) / n_r
    v["ir"] = (MS["ir"] - MS["itr,e"]) / n_t
    v["tr"] = (MS["tr"] - MS["itr,e"]) / n_i
    v["i"] = (MS["i"] - MS["it"] - MS["ir"] + MS["itr,e"]) / (n_t * n_r)
    v["t"] = (MS["t"] - MS["it"] - MS["tr"] + MS["itr,e"]) / (n_i * n_r)
    v["r"] = (MS["r"] - MS["ir"] - MS["tr"] + MS["itr,e"]) / (n_i * n_t)
    return {k: max(0.0, float(val)) for k, val in v.items()}


def format_sem(v: dict[str, float], n_items: int) -> float:
    """Standard error of a model's benchmark score that comes from picking ONE prompt template and
    ONE option order, for a benchmark of n_items items. The template/order main effects and their
    interaction do not shrink as items are added; the item interactions do."""
    var = v["t"] + v["r"] + v["tr"] + (v["it"] + v["ir"] + v["itr,e"]) / n_items
    return float(np.sqrt(var))


def mdd95(sem: float) -> float:
    """Minimal detectable difference between two scores each carrying error `sem`
    (MDC95 = 1.96 * sqrt(2) * SEM; Weir, 2005)."""
    return Z95 * np.sqrt(2) * sem


def facet_array(df: pd.DataFrame, model: str) -> tuple[np.ndarray, list, list]:
    f = df[(df["arm"] == "facets") & (df["model"] == model)]
    templates = sorted(f["template"].unique())
    rotations = sorted(f["rotation"].unique())
    piv = f.pivot_table(index="item", columns=["template", "rotation"], values="correct")
    piv = piv.dropna()  # keep only items observed in every condition (balanced design)
    y = piv.to_numpy(float).reshape(len(piv), len(templates), len(rotations))
    return y, templates, rotations


def g_study(df: pd.DataFrame, benchmark_size: int = 14042) -> pd.DataFrame:
    rows = []
    for model in sorted(df["model"].unique()):
        y, _, _ = facet_array(df, model)
        if y.size == 0:
            continue
        v = variance_components(y)
        total = sum(v.values()) or 1.0
        n = y.shape[0]
        rows.append({
            "model": model, "n_items": n,
            **{f"var_{k}": val for k, val in v.items()},
            **{f"pct_{k}": 100 * val / total for k, val in v.items()},
            "sem_format_sample": format_sem(v, n),
            "sem_format_full": format_sem(v, benchmark_size),
            "mdd95_sample": mdd95(format_sem(v, n)),
            "mdd95_full": mdd95(format_sem(v, benchmark_size)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- item-level consistency
def item_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: share of items whose correctness changes across the facet conditions, and share
    whose chosen CONTENT (original option) is the same under all 4 option orders, per template."""
    f = df[df["arm"] == "facets"]
    rows = []
    for model, g in f.groupby("model"):
        per_item = g.groupby("item")["correct"].agg(["min", "max"])
        flips = float((per_item["min"] != per_item["max"]).mean())
        same = []
        for _, gt in g.groupby("template"):
            oc = gt.groupby("item")["orig_choice"].agg(lambda s: s.nunique() == 1
                                                       and s.iloc[0] != "")
            same.append(oc.mean())
        rows.append({"model": model, "items_flipping": flips,
                     "order_invariant_items": float(np.mean(same))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- position bias
def position_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Correct answers are balanced across A-D by design, so an unbiased responder picks each
    letter 25% of the time. Chi-square test of the chosen-letter distribution against uniform;
    Cramer's V-style effect size w = sqrt(chi2 / n)."""
    f = df[(df["arm"] == "facets") & (df["parsed"] == 1)]
    rows = []
    for model, g in f.groupby("model"):
        counts = g["choice"].value_counts().reindex(list("ABCD"), fill_value=0)
        chi2, p = stats.chisquare(counts.to_numpy())
        n = counts.sum()
        acc_by_pos = g.groupby("correct_letter")["correct"].mean().reindex(list("ABCD"))
        rows.append({"model": model, **{f"pick_{k}": counts[k] / n for k in "ABCD"},
                     **{f"acc_when_{k}": acc_by_pos[k] for k in "ABCD"},
                     "chi2": chi2, "p": p, "w": float(np.sqrt(chi2 / n))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- test-retest
def fleiss_kappa(table: np.ndarray) -> float:
    """table: (n_subjects, n_categories) counts of raters choosing each category."""
    n = table.sum(1)
    if not np.all(n == n[0]) or n[0] < 2:
        raise ValueError("every subject needs the same number (>=2) of ratings")
    k = n[0]
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - k) / (k * (k - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def retest(df: pd.DataFrame) -> pd.DataFrame:
    """Sampled runs (temperature > 0) of the same prompt: agreement in the chosen option across
    seeds (Fleiss' kappa; unparsed answers form their own category) and SD of accuracy."""
    r = df[df["arm"] == "retest"]
    rows = []
    for model, g in r.groupby("model"):
        piv = g.pivot_table(index="item", columns="seed", values="orig_choice", aggfunc="first")
        piv = piv.dropna()
        cats = ["0", "1", "2", "3", ""]
        vals = piv.astype(str)
        table = np.stack([(vals == c).sum(axis=1).to_numpy() for c in cats], axis=1)
        accs = g.groupby("seed")["correct"].mean()
        rows.append({"model": model, "n_items": len(piv), "n_runs": piv.shape[1],
                     "fleiss_kappa": fleiss_kappa(table) if len(piv) else np.nan,
                     "all_runs_agree": float((table.max(1) == piv.shape[1]).mean()),
                     "acc_mean": accs.mean(), "acc_sd": accs.std(ddof=1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model rankings
def rank_flips(df: pd.DataFrame) -> pd.DataFrame:
    """For each model pair: in how many facet conditions does A beat B, and the overall difference
    (mean over conditions) with a paired item-bootstrap 95% CI."""
    acc = condition_accuracy(df)
    wide = acc.pivot_table(index=["template", "rotation"], columns="model", values="accuracy")
    f = df[df["arm"] == "facets"]
    item_means = f.groupby(["model", "item"])["correct"].mean().unstack(0).dropna()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(item_means), size=(2000, len(item_means)))
    rows = []
    for a, b in itertools.combinations(sorted(wide.columns), 2):
        diff_items = (item_means[a] - item_means[b]).to_numpy()
        boots = diff_items[idx].mean(1)
        rows.append({"a": a, "b": b, "diff": diff_items.mean(),
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                     "a_wins": int((wide[a] > wide[b]).sum()), "b_wins": int((wide[b] > wide[a]).sum()),
                     "ties": int((wide[a] == wide[b]).sum()), "n_conditions": len(wide),
                     "min_diff": float((wide[a] - wide[b]).min()),
                     "max_diff": float((wide[a] - wide[b]).max())})
    return pd.DataFrame(rows)


def parse_rates(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "arm", "template"])["parsed"].mean()
            .rename("parse_rate").reset_index())


# ---------------------------------------------------------------- pre-registered test of H1
def cochran_q(y: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for k related binary samples. y: (n_items, k) of 0/1. Tests whether the
    proportion correct differs across the k conditions."""
    y = np.asarray(y, float)
    k = y.shape[1]
    col, row = y.sum(0), y.sum(1)
    denom = k * row.sum() - (row ** 2).sum()
    if denom == 0:
        return 0.0, 1.0
    q = (k - 1) * (k * (col ** 2).sum() - row.sum() ** 2) / denom
    return float(q), float(stats.chi2.sf(q, k - 1))


def condition_tests(df: pd.DataFrame) -> pd.DataFrame:
    """H1: per model, Cochran's Q across all template x order conditions, Holm-corrected."""
    rows = []
    for model in sorted(df["model"].unique()):
        y, _, _ = facet_array(df, model)
        q, p = cochran_q(y.reshape(len(y), -1))
        rows.append({"model": model, "Q": q, "df": y.shape[1] * y.shape[2] - 1, "p": p})
    out = pd.DataFrame(rows)
    order = np.argsort(out["p"].to_numpy())
    m, running = len(out), 0.0
    holm = np.empty(m)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out["p"].iloc[i]))
        holm[i] = running
    out["p_holm"] = holm
    return out
