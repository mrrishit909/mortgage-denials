"""The project's check: fails loudly if the cleaning, the models, the audit or the privacy rules break.

    ./venv/bin/python check.py

1. the funnel recomputes; models use only allowlisted inputs (no race, ethnicity, sex, age, tract or post-decision fields)
2. refitting reproduces the 2024 AUCs, and an independent rank-based AUC agrees with scikit-learn's
3. group denial rates recompute from the data; Wilson intervals match statsmodels
4. the leakage demo holds; no published cell has fewer than 30 applications; no row-level data is written
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.proportion import proportion_confint

import analysis as A

R = Path(__file__).parent / "results"
FORBIDDEN = ("race", "ethnicity", "sex", "age", "tract", "ffiec", "census", "purchaser", "preapproval", "aus", "interest",
             "rate_spread", "denial", "credit_score", "payable")


def rank_auc(y, p):
    """Mann-Whitney: the chance a random denied application scores above a random approved one."""
    r = rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def main():
    d, steps = A.clean(A.load())
    assert [[s, n, round(r, 4)] for s, n, r in steps] == json.loads((R / "cleaning.json").read_text())
    used = set(A.LR_X) | set(A.GB_X) | {"lender"}
    assert used <= A.ALLOWED and not any(w in c for c in used for w in FORBIDDEN)

    y, te, preds, models, fc = A.fit_models(d)
    m = json.loads((R / "metrics.json").read_text())
    for k, p in preds.items():
        assert abs(rank_auc(y[te], p) - m[k]["auc"]) < 1e-6, k
    honest = m["gradient boosting"]["auc"]
    assert m["with leaky fields (wrong)"]["auc"] > 0.95 and honest < 0.9 and honest > m["logistic regression"]["auc"]

    g = A.group(d)[te].values
    audit = [a for a in json.loads((R / "audit.json").read_text()) if a["model"] == "gradient boosting"]
    for a in audit:
        mask = g == a["group"]
        k, n = int(y[te][mask].sum()), int(mask.sum())
        assert n == a["n"] and abs(k / n - a["denial_rate"]) < 1e-12
        lo, hi = proportion_confint(k, n, method="wilson")
        assert abs(lo - a["denial_ci"][0]) < 1e-9 and abs(hi - a["denial_ci"][1]) < 1e-9
    assert min(a["n"] for a in audit) >= A.MIN_CELL
    reasons = pd.read_csv(R / "denial_reasons.csv", index_col=0)
    assert (reasons["denials"] >= A.MIN_CELL).all()
    for f in R.iterdir():
        if f.suffix == ".csv":
            assert len(pd.read_csv(f)) < 100, f"{f.name} looks row-level"

    b = next(a for a in audit if a["group"] == "Black")
    w = next(a for a in audit if a["group"] == "White")
    print(f"OK: funnel recomputes ({len(d):,} applications); inputs allowlisted; 2024 AUC {honest:.3f} reproduced by rank formula "
          f"(leaky model {m['with leaky fields (wrong)']['auc']:.3f}); group rates and Wilson CIs recompute "
          f"(Black {100 * b['denial_rate']:.1f}% vs White {100 * w['denial_rate']:.1f}%); all published cells >= {A.MIN_CELL}")


if __name__ == "__main__":
    main()
