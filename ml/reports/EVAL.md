# Pitstop Oracle — walk-forward evaluation

Training data range: **2022-03-20 to 2026-11-01**.
Walk-forward folds train and calibrate only on races before the test Grand Prix.
Primary scores use the same DNF-adjusted Plackett–Luce Monte Carlo path as inference.
Pre-quali baseline: shifted car/driver form ratings. Post-quali baselines: official grid and qualifying gaps.
No production model artifacts are written by this evaluation run.

## Dataset

- Rows: 2227
- Races: 111
- Pole win rate: **60.4%**

## 2026 walk-forward (new regulation era)

- Races scored: 19
- Date range: 2026-03-08 to 2026-11-01
- Pre-quali hit / top-3 / log-loss / Brier: **21.1% / 52.6% / 2.085 / 0.040**
- Prior-form baseline log-loss: **2.275**
- Post-quali hit / top-3 / log-loss / Brier: **42.1% / 78.9% / 1.618 / 0.038**
- Post-quali rank correlation / finish MAE: **0.697 / 3.151**
- Grid / quali-gap log-loss: **1.609 / 1.559**
- DNF log-loss / Brier / PR-AUC / ROC-AUC: **0.503 / 0.158 / 0.300 / 0.508**
- Post-sprint races / win hit / top-3 / log-loss / Brier: **6 / 16.7% / 83.3% / 1.311 / 0.033**

- 2026 R01: post hit 100.0%, post LL 1.296, pole LL 0.000, quali-gap LL 0.886
- 2026 R02: post hit 0.0%, post LL 1.898, pole LL 0.000, quali-gap LL 1.242
- 2026 R03: post hit 0.0%, post LL 1.063, pole LL 0.000, quali-gap LL 1.222
- 2026 R04: post hit 100.0%, post LL 0.329, pole LL 0.000, quali-gap LL 1.431
- 2026 R05: post hit 0.0%, post LL 1.760, pole LL 34.539, quali-gap LL 1.882
- 2026 R06: post hit 100.0%, post LL 0.553, pole LL 0.000, quali-gap LL 1.585
- 2026 R07: post hit 100.0%, post LL 0.704, pole LL 34.539, quali-gap LL 1.734
- 2026 R08: post hit 0.0%, post LL 2.425, pole LL 0.000, quali-gap LL 1.652
- 2026 R09: post hit 0.0%, post LL 3.994, pole LL 34.539, quali-gap LL 1.688
- 2026 R10: post hit 0.0%, post LL 2.050, pole LL 0.000, quali-gap LL 1.310
- 2026 R11: post hit 0.0%, post LL 2.058, pole LL 0.000, quali-gap LL 1.577
- 2026 R12: post hit 0.0%, post LL 1.533, pole LL 0.000, quali-gap LL 1.716
- 2026 R13: post hit 0.0%, post LL 3.250, pole LL 34.539, quali-gap LL 2.546
- 2026 R14: post hit 0.0%, post LL 1.262, pole LL 34.539, quali-gap LL 1.771
- 2026 R15: post hit 100.0%, post LL 0.733, pole LL 0.000, quali-gap LL 0.616
- 2026 R16: post hit 100.0%, post LL 0.663, pole LL 0.000, quali-gap LL 1.278
- 2026 R17: post hit 100.0%, post LL 0.919, pole LL 0.000, quali-gap LL 1.662
- 2026 R18: post hit 100.0%, post LL 1.074, pole LL 0.000, quali-gap LL 1.575
- 2026 R19: post hit 0.0%, post LL 3.174, pole LL 34.539, quali-gap LL 2.257

## Transfer diagnostic (train <2026, test 2026 only)

- Train races: 92
- Test races (2026): 19
- Post-quali hit: **42.1%**
- Post-quali log-loss: **1.367**
- Quali-gap log-loss: **1.559**

## When model pick ≠ pole (train set)

- Races: 39
- Model hit: **43.6%**
- Log-loss: **1.096**


## All-season chronological out-of-sample window

- Test races: 91 (2022-11-13 to 2026-11-01)
- Every eligible race is held out once; each fit and calibration set contains earlier races only.
- Seasons: 2022, 2023, 2024, 2025, 2026
- Pre-quali races / hit / log-loss / prior-form log-loss: 91 / 37.4% / 1.788 / 2.386
- Pre-quali top-3 / Brier / rank correlation / finish MAE: 70.3% / 0.037 / 0.615 / 3.492
- Post-quali races / hit / log-loss / grid log-loss / quali-gap log-loss: 91 / 48.4% / 1.422 / 1.466 / 1.911
- Post-quali top-3 / Brier / rank correlation / finish MAE: 84.6% / 0.033 / 0.676 / 3.101
- DNF log-loss / Brier / PR-AUC / ROC-AUC: 0.391 / 0.115 / 0.280 / 0.552
- Post-sprint races / win hit / top-3 / log-loss / Brier: 12 / 33.3% / 91.7% / 1.345 / 0.034

- 2022: 2 races; pre LL 2.003; post LL 1.420; DNF PR-AUC 0.157
- 2023: 22 races; pre LL 1.199; post LL 1.209; DNF PR-AUC 0.261
- 2024: 24 races; pre LL 2.004; post LL 1.708; DNF PR-AUC 0.316
- 2025: 24 races; pre LL 1.860; post LL 1.178; DNF PR-AUC 0.257
- 2026: 19 races; pre LL 2.085; post LL 1.618; DNF PR-AUC 0.300

Training-set diagnostics are stored separately under `train_*` keys in `metrics.json`; they are not performance claims.
Win probability temperature is calibrated from preceding races in each fold; DNF probabilities are reported separately from conditional classified finishing distributions.

Important limitation: historical practice and weather snapshots are unavailable at forecast cutoffs, so current-event FP and weather features are excluded from both model feature views.
