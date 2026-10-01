# Who gets denied a mortgage in Tampa Bay? (predictive model + fairness audit)

A model that predicts which home-purchase mortgage applications lenders deny. It uses public HMDA data for Hillsborough,
Pinellas, Pasco and Hernando counties: trained on 2022–2023, tested on 2024. The model is then audited by race and
ethnicity, which are never model inputs.

Build log: https://mrrishit909.github.io/projects/mortgage-denials/

**What it is for:** helping housing counsellors decide whom to help *before* they apply. **What it is not:** an
underwriting tool. The target is the lender's past decision, not whether a borrower would repay, so the model learns how
lenders behaved, including any bias. Public HMDA also has no credit scores.

## Data

CFPB/FFIEC Home Mortgage Disclosure Act (HMDA) loan/application register, via the HMDA Data Browser API:
home-purchase applications that were approved (originated, or approved but not taken up) or denied, 2022–2024.
The public file is already privacy-modified by the CFPB (ages in ranges, amounts rounded). This repo publishes only
aggregates; no row-level data is committed.

## Steps

| Step | File | What it does |
|---|---|---|
| 1 | `download.py` | 177,971 decisions (by county, because the metro codes changed in 2024) |
| 2 | `analysis.py` | funnel; allowlisted features; two models; leakage demo; calibration; importance; group audit |
| 3 | `charts.py` | five charts |
| 4 | `check.py` | funnel recomputes; inputs allowlisted; AUC reproduced with an independent rank formula; Wilson CIs; privacy rules |

## Cleaning (denial rate at each step, since dropping rows can shift it)

| Rule | Left | Denied |
|---|---:|---:|
| Home-purchase decisions, 2022–2024 | 177,971 | 14.8% |
| First lien, owner's main home | 148,621 | 14.2% |
| Site-built 1–4 units (manufactured homes are a separate market, denied far more often) | 138,225 | 11.6% |
| Not business, reverse or open-end | 136,615 | 11.2% |
| Debt-to-income reported | 135,345 | 11.1% |
| Income $10k–$2M | 133,252 | 10.9% |
| Property value $50k–$5M | 132,647 | 10.6% |
| Loan term reported | 132,397 | 10.6% |

## The leakage trap

Many HMDA fields are filled in *after* the decision. Their presence alone gives the answer away:
- purchaser type: denied loans are never sold;
- preapproval: denied preapproval requests are filed under a different action code;
- interest rate and costs: missing for every denial;
- automated-underwriting and "initially payable" codes, and whether LTV was reported.

Adding them lifts the 2024 AUC from 0.80 to **0.975**, a model that looks nearly perfect and is useless before a
decision exists.

The fix is an **allowlist** of facts known at application: income, loan amount, loan-to-value (computed from loan
amount and property value), debt-to-income, loan type, term, conforming limit, application channel, units, county
and (in one variant) lender. Excluded:
- race, ethnicity, sex and age (protected under the Equal Credit Opportunity Act);
- census-tract fields such as minority population share, which proxy for race.

## Results (2024 applications, n = 41,441, 10.4% denied)

| Model | AUC | PR-AUC | Brier | Top 20% flagged: share of denials caught | Precision |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.761 | 0.504 | 0.069 | 57.5% | 30.0% |
| Gradient boosting | **0.802** | **0.577** | **0.062** | **61.6%** | **32.1%** |
| Gradient boosting + lender | 0.825 | 0.603 | 0.060 | 63.9% | 33.3% |
| + post-decision fields (wrong) | 0.975 | 0.838 | 0.037 | 94.4% | 49.2% |

- **Calibration.** Gradient boosting is well calibrated on 2024: in the riskiest tenth it predicted 49% and 53% were
  denied. Logistic regression under-predicts at the low end.
- **What drives predictions** (drop in AUC when a feature is shuffled): debt-to-income 0.193, loan type 0.050,
  loan-to-value 0.037, loan amount 0.036, channel 0.013, income 0.009.
- **Drift.** The denial rate held steady (10.7%, 10.7%, 10.4%), so training on 2022–23 transfers to 2024.

## Audit by group (gradient boosting; groups are audit labels only)

| Group (2024) | n | Denied | Model predicted | Denials caught in top 20% |
|---|---:|---:|---:|---:|
| White | 20,278 | 8.4% | 9.6% | 57% |
| Hispanic | 9,446 | 13.7% | 13.0% | 65% |
| Black | 2,902 | 15.5% | 13.4% | 66% |
| Asian | 2,611 | 7.7% | 8.9% | 58% |
| Other or joint | 1,208 | 8.9% | 10.7% | 69% |
| Race not reported | 4,996 | 11.3% | 11.2% | 63% |

- **For counselling triage, recall is the metric that matters:** does the model miss one group's denials more? It
  doesn't. It catches 66% of Black and 65% of Hispanic applicants' denials, against 57% of White applicants'.
- **But calibration differs.** Black and Hispanic applicants were denied more often than the model's inputs predict,
  and White and Asian applicants less often. At the same model score, the odds of denial vs White applicants are:
  - Black: **1.54** (95% CI 1.34–1.77);
  - Hispanic: **1.31** (1.19–1.44);
  - race not reported: 1.25 (1.10–1.41);
  - Asian: 0.97, not different.

  Adding the lender as a feature does not shrink these gaps (Black 1.59, Hispanic 1.35). So they aren't explained by
  which lenders people apply to.
- **This is not proof of discrimination.** Public HMDA has no credit scores, and lenders cite credit history as the
  main reason for 14% of Black applicants' denials vs 9% of White applicants'. A gap at the same inputs is what a
  fair-lending review would investigate with the full lender file.
- With different base rates, calibration and equal error rates across groups can't all hold at once. So the model is
  not labelled "fair" or "unfair" on any one number.

**Main denial reason** (2022–2024, share of each group's denials):

| Reason | White | Hispanic | Black | Asian |
|---|---:|---:|---:|---:|
| Debt-to-income | 38% | 47% | 48% | 49% |
| Collateral (appraisal) | 23% | 14% | 10% | 10% |
| Credit history | 9% | 9% | 14% | 5% |
| Incomplete application | 9% | 6% | 6% | 8% |

## Not done

- No credit scores, assets or employment detail (not in public HMDA).
- Withdrawn and incomplete-file applications are left out: the target is approved vs denied.
- Manufactured homes, refinances and other metros are not modelled.
- No cell with fewer than 30 applications is published.

## Run it

```
python3 -m venv venv && ./venv/bin/pip install pandas numpy statsmodels scikit-learn matplotlib scipy
./venv/bin/python download.py && ./venv/bin/python analysis.py && ./venv/bin/python charts.py && ./venv/bin/python check.py
```
