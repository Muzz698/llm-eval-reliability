import numpy as np
import pandas as pd
import pytest

from evalrel import metrics as M
from evalrel.items import rotate
from evalrel.prompts import TEMPLATES, parse_letter
from evalrel.runner import conditions, run

DESIGN = {"templates": ["plain", "mmlu", "paren"], "rotations": [0, 1, 2, 3],
          "retest": {"template": "plain", "rotation": 0, "temperature": 0.7, "seeds": [1, 2, 3]}}


def make_items(n=40):
    return [{"id": f"s_{i}", "subject": "high_school_biology", "question": f"Q{i}?",
             "choices": [f"opt{i}_{j}" for j in range(4)], "answer": i % 4} for i in range(n)]


def test_rotation_counterbalances_the_correct_position():
    item = make_items(1)[0]
    positions = set()
    for k in range(4):
        choices, ans = rotate(item, k)
        assert choices[ans] == item["choices"][item["answer"]]
        positions.add(ans)
    assert positions == {0, 1, 2, 3}


@pytest.mark.parametrize("text,expected", [
    ("B", "B"), (" (C)", "C"), ("D.", "D"), ("A) Paris", "A"), ("The answer is B", "B"),
    ("Answer: (D)", "D"), ("**B**", "B"), ("The correct answer is:\n\n**C.", "C"), ("C\nBecause", "C"), ("A cell is a unit", None), ("I am not sure", None), ("", None),
])
def test_parse_letter(text, expected):
    assert parse_letter(text) == expected


def test_templates_show_all_options():
    item = make_items(1)[0]
    for t in TEMPLATES.values():
        prompt = t(item, item["choices"])
        assert all(c in prompt for c in item["choices"]) and item["question"] in prompt


class FakeClient:
    """Answers by content: always correct on even items; on odd items always picks position A."""
    model = "fake"

    def __init__(self, items):
        self.lookup = {it["question"]: it for it in items}

    def generate(self, prompt, temperature=0.0, seed=0, max_tokens=8):
        item = next(it for q, it in self.lookup.items() if q in prompt)
        if int(item["id"].split("_")[1]) % 2 == 0:
            correct = item["choices"][item["answer"]]
            for line in prompt.splitlines():
                if line.endswith(" " + correct):
                    return line.strip("(")[0]
        return "A"


def test_run_is_complete_and_resumable(tmp_path):
    items = make_items(12)
    out = tmp_path / "fake.csv"
    run(FakeClient(items), items, DESIGN, out)
    df = M.load_results([out])
    assert len(df) == len(items) * len(conditions(DESIGN))
    run(FakeClient(items), items, DESIGN, out)  # second call adds nothing
    assert len(M.load_results([out])) == len(df)
    assert df["parsed"].all()


def test_metrics_on_fake_responder(tmp_path):
    items = make_items(40)
    out = tmp_path / "fake.csv"
    run(FakeClient(items), items, DESIGN, out)
    df = M.load_results([out])
    # even items always right; odd items right only when the correct answer is in position A
    spread = M.accuracy_spread(df).iloc[0]
    assert spread["mean"] == pytest.approx(0.5 + 0.5 * 0.25)
    pb = M.position_bias(df).iloc[0]
    assert pb["pick_A"] > 0.5 and pb["p"] < 1e-6
    cons = M.item_consistency(df).iloc[0]
    assert cons["order_invariant_items"] == pytest.approx(0.5)
    rt = M.retest(df).iloc[0]
    assert rt["fleiss_kappa"] == pytest.approx(1.0)


def test_variance_components_recover_known_structure():
    rng = np.random.default_rng(1)
    n_i, n_t, n_r = 400, 3, 4
    item = rng.normal(0, 1.0, n_i)[:, None, None]
    order = np.array([0.0, 0.5, -0.5, 0.0])[None, None, :]
    noise = rng.normal(0, 0.3, (n_i, n_t, n_r))
    v = M.variance_components(item + order + noise)
    assert v["i"] == pytest.approx(1.0, rel=0.2)
    assert v["r"] == pytest.approx(np.var(order, ddof=1), rel=0.1)
    assert v["t"] < 0.01 and v["itr,e"] == pytest.approx(0.09, rel=0.2)


def test_fleiss_kappa_bounds():
    perfect = np.array([[3, 0], [0, 3], [3, 0]])
    assert M.fleiss_kappa(perfect) == pytest.approx(1.0)
    split = np.array([[2, 1], [1, 2], [2, 1], [1, 2]])
    assert M.fleiss_kappa(split) < 0


def test_rank_flips_detects_reversal():
    rows = []
    for model, accs in {"m1": [0.6, 0.6, 0.4], "m2": [0.5, 0.5, 0.5]}.items():
        for c, acc in enumerate(accs):
            for i in range(10):
                rows.append({"model": model, "arm": "facets", "item": f"i{i}", "template": "plain",
                             "rotation": c, "correct": int(i < acc * 10)})
    rf = M.rank_flips(pd.DataFrame(rows)).iloc[0]
    assert (rf["a_wins"], rf["b_wins"]) == (2, 1)


def test_mdd_formula():
    assert M.mdd95(1.0) == pytest.approx(2.772, abs=1e-3)


def test_cochran_q_detects_condition_effect():
    rng = np.random.default_rng(2)
    same = rng.integers(0, 2, (200, 1)).repeat(4, axis=1)
    assert M.cochran_q(same)[1] == 1.0 or M.cochran_q(same)[0] == 0
    shifted = same.copy()
    shifted[:80, 3] = 1 - shifted[:80, 3] * 0  # condition 4 right on 80 more items
    assert M.cochran_q(shifted)[1] < 1e-6
