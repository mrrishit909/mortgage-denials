"""Step 2: predict which Tampa Bay home-purchase applications get denied, then audit the model by group.

    ./venv/bin/python analysis.py

The target is the LENDER'S decision (denied vs approved), not whether the borrower would repay, so the model learns how
lenders behaved. It is a triage tool for housing counsellors (who to help before applying), never an underwriting tool.
Features come from an allowlist of facts known when someone applies. Race, ethnicity, sex and age, and neighbourhood
fields that proxy for them, are left out, then used only to audit. Train on 2022-23, test on 2024.
Only aggregates are written; no row-level data leaves data/raw/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
TRAIN, TEST = (2022, 2023), 2024
NUMERIC = ["log_income", "log_loan", "ltv", "dti"]
CATEGORICAL = ["dti_band", "loan_type", "term", "conforming", "channel", "units", "county"]
ALLOWED = set(NUMERIC + CATEGORICAL + ["lender"])          # check.py asserts every model uses only these
LR_X = NUMERIC[:3] + CATEGORICAL                           # logistic regression: DTI as bands, so its shape is free
GB_X = NUMERIC + [c for c in CATEGORICAL if c != "dti_band"]   # boosting: exact DTI; trees find the shape themselves
LEAKY = ["purchaser_type", "preapproval", "aus-1", "initially_payable_to_institution", "applicant_credit_score_type", "price_missing", "ltv_missing"]
FLAG_SHARE = 0.20                                          # counsellors can see the riskiest fifth of applicants
MIN_CELL = 30                                              # smaller published cells are suppressed
REASONS = {"1": "debt-to-income", "2": "employment history", "3": "credit history", "4": "collateral", "5": "insufficient cash",
           "6": "unverifiable information", "7": "incomplete application", "8": "mortgage insurance denied", "9": "other"}


def load():
    return pd.concat([pd.read_csv(HERE / "data" / "raw" / f"hmda_{y}.csv", dtype=str, low_memory=False) for y in (*TRAIN, TEST)],
                     ignore_index=True)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def clean(d):
    """Documented rules; each step's count and denial rate are recorded."""
    d = d.assign(denied=(d["action_taken"] == "3").astype(int))
    steps = [("home-purchase decisions, 2022-2024", len(d), d["denied"].mean())]
    inc, val, amt, term = num(d["income"]), num(d["property_value"]), num(d["loan_amount"]), num(d["loan_term"])
    rules = [
        ("first lien, owner's main home", (d["lien_status"] == "1") & (d["occupancy_type"] == "1")),
        ("site-built 1-4 units (manufactured homes are a separate market)", d["derived_dwelling_category"] == "Single Family (1-4 Units):Site-Built"),
        ("not business, reverse or open-end", (d["business_or_commercial_purpose"] == "2") & (d["reverse_mortgage"] == "2") & (d["open-end_line_of_credit"] == "2")),
        ("debt-to-income reported", ~d["debt_to_income_ratio"].isin(["NA", "Exempt"]) & d["debt_to_income_ratio"].notna()),
        ("income $10k-$2M reported", inc.between(10, 2000)),
        ("property value $50k-$5M reported", val.between(50_000, 5_000_000)),
        ("loan term reported", term.between(60, 480)),
    ]
    keep = pd.Series(True, index=d.index)
    for name, cond in rules:
        keep &= cond.fillna(False)
        steps.append((name, int(keep.sum()), float(d.loc[keep, "denied"].mean())))
    return d[keep].copy(), steps


def dti_value(band):
    """HMDA gives exact DTI for 36-49% and ranges elsewhere: use each range's middle."""
    mid = {"<20%": 15, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}
    return band.map(mid).fillna(num(band.str.rstrip("%")))


def features(d, lenders=None):
    f = pd.DataFrame(index=d.index)
    f["log_income"] = np.log(num(d["income"]) * 1000)
    f["log_loan"] = np.log(num(d["loan_amount"]))
    f["ltv"] = (num(d["loan_amount"]) / num(d["property_value"])).clip(0, 1.5)    # own LTV: both known at application
    f["dti"] = dti_value(d["debt_to_income_ratio"])
    f["dti_band"] = pd.cut(f["dti"], [0, 20, 30, 36, 43, 50, 60, 100], right=False).astype(str)
    f["loan_type"] = d["loan_type"].map({"1": "conventional", "2": "FHA", "3": "VA", "4": "USDA"})
    f["term"] = np.where(num(d["loan_term"]) == 360, "30-year", "other")
    f["conforming"] = d["conforming_loan_limit"].fillna("NA")
    f["channel"] = d["submission_of_application"].map({"1": "direct", "2": "via another lender"}).fillna("other")
    f["units"] = np.where(d["total_units"] == "1", "1", "2-4")
    f["county"] = d["county_code"].map({"12057": "Hillsborough", "12103": "Pinellas", "12101": "Pasco", "12053": "Hernando"})
    if lenders is not None:
        f["lender"] = d["lei"].where(d["lei"].isin(lenders), "other")
    return f


def group(d):
    """Audit groups only, never model inputs. Hispanic of any race first, then race; tiny groups pooled."""
    g = pd.Series("Other or joint", index=d.index)
    g[d["derived_race"] == "Race Not Available"] = "Not reported"
    g[d["derived_race"] == "Asian"] = "Asian"
    g[d["derived_race"] == "Black or African American"] = "Black"
    g[d["derived_race"] == "White"] = "White"
    g[d["derived_ethnicity"] == "Hispanic or Latino"] = "Hispanic"
    return g


def leaky_features(d):
    f = pd.DataFrame(index=d.index)
    for c in ["purchaser_type", "preapproval", "aus-1", "initially_payable_to_institution", "applicant_credit_score_type"]:
        f[c] = d[c].fillna("NA")
    f["price_missing"] = d["interest_rate"].isin(["NA", "Exempt"]).astype(str)
    f["ltv_missing"] = d["loan_to_value_ratio"].isin(["NA", "Exempt"]).astype(str)
    return f


def gbm():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, categorical_features="from_dtype",
                                          early_stopping=True, validation_fraction=0.15, random_state=0)


def as_cat(f):
    """Text columns -> pandas categories. Convert the WHOLE frame once, so train and test share category codes."""
    f = f.copy()
    for c in f.columns:
        if not pd.api.types.is_numeric_dtype(f[c]):
            f[c] = f[c].astype("category")
    return f


def logit():
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC[:3]),            # DTI enters as bands, so its shape is free
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def evaluate(y, p):
    cut = np.quantile(p, 1 - FLAG_SHARE)
    flag = p >= cut
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "base_rate": float(np.mean(y)), "mean_predicted": float(np.mean(p)),
            "top20_recall": float(flag[y == 1].mean()), "top20_precision": float(y[flag].mean()), "threshold": float(cut)}


def wilson(k, n, z=1.959963984540054):        # z for a 95% interval
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def audit(y, p, g, cut):
    """Per group: who gets denied, what the model expected, and whether it catches that group's denials (recall)."""
    rows = []
    flag = p >= cut
    for name in ["White", "Hispanic", "Black", "Asian", "Other or joint", "Not reported"]:
        m = np.asarray(g == name)
        n, k = int(m.sum()), int(y[m].sum())
        if n < MIN_CELL or k < MIN_CELL:
            rows.append({"group": name, "n": n, "suppressed": True})
            continue
        caught = int(flag[m & (y == 1)].sum())
        rows.append({"group": name, "n": n, "denied": k, "denial_rate": k / n, "denial_ci": wilson(k, n),
                     "mean_predicted": float(p[m].mean()), "observed_over_predicted": (k / n) / float(p[m].mean()),
                     "flag_rate": float(flag[m].mean()), "recall": caught / k, "recall_ci": wilson(caught, k),
                     "false_flag_rate": float(flag[m & (y == 0)].mean())})
    return rows


def gap_after_inputs(y, p, g):
    """Odds of denial vs White applicants with the same model score: 'denied more often than the inputs predict'."""
    lp = np.log(p / (1 - p))
    X = pd.get_dummies(g, dtype=float).drop(columns="White")
    X.insert(0, "score", lp)
    m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    ci = m.conf_int()
    return {c: {"odds_ratio": float(np.exp(m.params[c])), "lo": float(np.exp(ci.loc[c, 0])), "hi": float(np.exp(ci.loc[c, 1]))}
            for c in X.columns if c != "score"}


def fit_models(d):
    """Train on 2022-23, predict 2024. Returns labels, the test mask, predictions, fitted models and the feature frame."""
    yr = num(d["activity_year"])
    tr, te = yr.isin(TRAIN).values, (yr == TEST).values
    y = d["denied"].values
    top = d.loc[tr, "lei"].value_counts().index[:60]                    # the 60 biggest lenders in the training years
    f = features(d, lenders=top)
    assert set(f.columns) <= ALLOWED
    preds = {}
    lr = logit().fit(f.loc[tr, LR_X], y[tr])
    preds["logistic regression"] = lr.predict_proba(f.loc[te, LR_X])[:, 1]
    fc = as_cat(f)
    gb = gbm().fit(fc.loc[tr, GB_X], y[tr])
    preds["gradient boosting"] = gb.predict_proba(fc.loc[te, GB_X])[:, 1]
    gbl = gbm().fit(fc.loc[tr, GB_X + ["lender"]], y[tr])
    preds["gradient boosting + lender"] = gbl.predict_proba(fc.loc[te, GB_X + ["lender"]])[:, 1]
    lk = as_cat(pd.concat([f[GB_X], leaky_features(d)], axis=1))
    gbk = gbm().fit(lk.loc[tr], y[tr])
    preds["with leaky fields (wrong)"] = gbk.predict_proba(lk.loc[te])[:, 1]
    return y, te, preds, {"logistic regression": lr, "gradient boosting": gb}, fc


def main():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    raw = load()
    d, steps = clean(raw)
    (out / "cleaning.json").write_text(json.dumps([(s, n, round(r, 4)) for s, n, r in steps], indent=1) + "\n")
    for s, n, r in steps:
        print(f"{n:8,}  {100 * r:5.1f}% denied  {s}")

    y, te, preds, models, fc = fit_models(d)
    g = group(d)
    metrics = {k: evaluate(y[te], p) for k, p in preds.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(pd.DataFrame(metrics).T.round(3).to_string())

    cal = []
    for k in ["logistic regression", "gradient boosting"]:
        b = pd.qcut(preds[k], 10, labels=False, duplicates="drop")
        for i in range(int(b.max()) + 1):
            cal.append({"model": k, "decile": i + 1, "predicted": float(preds[k][b == i].mean()), "observed": float(y[te][b == i].mean()),
                        "n": int((b == i).sum())})
    pd.DataFrame(cal).round(4).to_csv(out / "calibration.csv", index=False)

    imp = permutation_importance(models["gradient boosting"], fc.loc[te, GB_X], y[te], scoring="roc_auc", n_repeats=5, random_state=0)
    pd.DataFrame({"feature": GB_X, "auc_drop": imp.importances_mean, "sd": imp.importances_std}).sort_values(
        "auc_drop", ascending=False).round(4).to_csv(out / "importance.csv", index=False)

    gt = g[te].values
    aud, gaps = [], {}
    for k in ["gradient boosting", "gradient boosting + lender"]:
        for r in audit(y[te], preds[k], gt, metrics[k]["threshold"]):
            aud.append({"model": k, **r})
        gaps[k] = gap_after_inputs(y[te], preds[k], pd.Series(gt))
    (out / "audit.json").write_text(json.dumps(aud, indent=1, default=float) + "\n")
    (out / "gap_after_inputs.json").write_text(json.dumps(gaps, indent=2) + "\n")
    print(pd.DataFrame([a for a in aud if not a.get("suppressed")]).drop(columns=["denial_ci", "recall_ci"]).round(3).to_string(index=False))
    print(json.dumps(gaps, indent=1))

    den = d[d["denied"] == 1]
    rs = pd.crosstab(group(den), den["denial_reason-1"].map(REASONS).fillna("not given"))
    rs = rs[rs.sum(axis=1) >= MIN_CELL]
    share = rs.div(rs.sum(axis=1), axis=0)
    share.insert(0, "denials", rs.sum(axis=1))
    share.round(4).to_csv(out / "denial_reasons.csv")
    print(share.round(3).to_string())

    d.groupby(features(d)["loan_type"])["denied"].agg(["size", "mean"]).round(4).to_csv(out / "denial_rate_by_loan_type.csv")
    by_year = d.groupby("activity_year")["denied"].agg(["size", "mean"]).round(4)
    by_year.to_csv(out / "denial_rate_by_year.csv")
    print(by_year)


if __name__ == "__main__":
    main()
