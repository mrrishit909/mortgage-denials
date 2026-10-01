"""Step 3: charts (aggregates only).

    ./venv/bin/python charts.py   -> charts/01_leakage.png ... 05_reasons.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.titlepad": 12,
})
GROUPS = ["White", "Hispanic", "Black", "Asian", "Other or joint", "Not reported"]
FEATURE_NAMES = {"dti_band": "debt-to-income band", "loan_type": "loan type (FHA, VA...)", "ltv": "loan-to-value",
                 "log_loan": "loan amount", "channel": "applied directly or via broker", "log_income": "income", "term": "30-year or other",
                 "county": "county", "dti": "debt-to-income", "conforming": "conforming limit", "units": "1 or 2-4 units"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / "charts" / name, dpi=150)
    plt.close(fig)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    m = json.loads((R / "metrics.json").read_text())
    names = ["logistic regression", "gradient boosting", "gradient boosting + lender", "with leaky fields (wrong)"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, k in enumerate(names):
        wrong = "wrong" in k
        ax.barh(i, m[k]["auc"], color=GRAY if wrong else BLUE, height=0.55, hatch="//" if wrong else None, edgecolor=BG)
        ax.text(m[k]["auc"] + 0.005, i, f"{m[k]['auc']:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(4), ["logistic regression", "gradient boosting", "gradient boosting + lender",
                             "+ fields filled in after\nthe decision (leak)"])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC on 2024 applications (trained on 2022-23); 0.5 = coin flip, 1 = perfect")
    ax.set_title("Fields recorded after the decision make a model look nearly perfect")
    save(fig, "01_leakage.png")

    c = pd.read_csv(R / "calibration.csv")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, color in [("logistic regression", ORANGE), ("gradient boosting", BLUE)]:
        s = c[c["model"] == k]
        ax.plot(100 * s["predicted"], 100 * s["observed"], color=color, marker="o", linewidth=2, markersize=5, label=k)
    lim = 100 * c[["predicted", "observed"]].max().max() * 1.05
    ax.plot([0, lim], [0, lim], color=DIM, linewidth=1, linestyle="--", label="perfectly calibrated")
    ax.set_xlabel("predicted denial rate (%), 10 equal groups of 2024 applicants")
    ax.set_ylabel("actual denial rate (%)")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    ax.set_title("Predicted risk tracks what happened in 2024 (each dot is a tenth of applicants)")
    save(fig, "02_calibration.png")

    imp = pd.read_csv(R / "importance.csv").sort_values("auc_drop")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh([FEATURE_NAMES[f] for f in imp["feature"]], imp["auc_drop"], color=BLUE, height=0.6, xerr=imp["sd"], ecolor=DIM)
    for i, v in enumerate(imp["auc_drop"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("drop in AUC when the feature is shuffled (2024 applications)")
    ax.set_title("Debt-to-income drives the predictions; loan type and loan-to-value follow")
    save(fig, "03_importance.png")

    a = pd.DataFrame(json.loads((R / "audit.json").read_text()))
    a = a[a["model"] == "gradient boosting"].set_index("group").loc[GROUPS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    y = np.arange(len(GROUPS))
    ax = axes[0]
    ax.hlines(y, 100 * a["mean_predicted"], 100 * a["denial_rate"], color=DIM, linewidth=2)
    ax.scatter(100 * a["mean_predicted"], y, color=GRAY, s=45, zorder=3, label="model predicted")
    ax.scatter(100 * a["denial_rate"], y, color=ORANGE, s=45, zorder=3, label="actually denied")
    ax.set_yticks(y, [f"{g} (n={int(n):,})" for g, n in zip(GROUPS, a["n"])])
    ax.invert_yaxis()
    ax.set_xlabel("denial rate, 2024 (%)")
    ax.legend(frameon=False, labelcolor=INK, loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title("Denied vs what the model expected", fontsize=12)
    ax = axes[1]
    lo = np.array([r[0] for r in a["recall_ci"]]); hi = np.array([r[1] for r in a["recall_ci"]])
    ax.hlines(y, 100 * lo, 100 * hi, color=BLUE, linewidth=2, alpha=0.6)
    ax.scatter(100 * a["recall"], y, color=BLUE, s=45, zorder=3)
    for i, v in enumerate(a["recall"]):
        ax.text(100 * hi[i] + 1, i, f"{100 * v:.0f}%", va="center", color=INK, fontsize=9)
    ax.set_xlim(45, 85)
    ax.set_xlabel("share of the group's denials flagged in the top 20% (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Who the model catches for counselling", fontsize=12)
    fig.suptitle("Audit by group (race and ethnicity are never model inputs)", x=0.01, ha="left", fontsize=13, color=INK)
    save(fig, "04_audit.png")

    r = pd.read_csv(R / "denial_reasons.csv", index_col=0).loc[GROUPS]
    reasons = ["debt-to-income", "collateral", "credit history", "incomplete application"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True)
    for ax, reason in zip(axes, reasons):
        ax.barh(y, 100 * r[reason], color=BLUE, height=0.6)
        for i, v in enumerate(r[reason]):
            ax.text(100 * v + 1, i, f"{100 * v:.0f}%", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, 100 * r[reasons].max().max() * 1.35)
        ax.set_title(reason, fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, GROUPS)
    axes[0].invert_yaxis()
    fig.suptitle("Main denial reason given by lenders, share of each group's denials (2022-2024)", x=0.01, ha="left", fontsize=13, color=INK)
    save(fig, "05_reasons.png")
    print("charts written")


if __name__ == "__main__":
    main()
