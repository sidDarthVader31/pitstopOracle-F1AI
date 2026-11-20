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
- Pre-quali hit / top-3 / log-loss / Brier: **26.3% / 68.4% / 1.948 / 0.039**
- Prior-form baseline log-loss: **2.292**
- Post-quali hit / top-3 / log-loss / Brier: **52.6% / 84.2% / 1.192 / 0.027**
- Post-quali rank correlation / finish MAE: **0.681 / 3.265**
- Grid / quali-gap log-loss: **1.617 / 1.576**
- DNF log-loss / Brier / PR-AUC / ROC-AUC: **0.472 / 0.149 / 0.392 / 0.634**
- Post-sprint races / win hit / top-3 / log-loss / Brier: **6 / 0.0% / 50.0% / 1.784 / 0.040**

- 2026 R01: post hit 100.0%, post LL 0.520, pole LL 0.000, quali-gap LL 0.968
- 2026 R02: post hit 0.0%, post LL 1.034, pole LL 0.000, quali-gap LL 1.436
- 2026 R03: post hit 100.0%, post LL 0.609, pole LL 0.000, quali-gap LL 1.222
- 2026 R04: post hit 100.0%, post LL 0.411, pole LL 0.000, quali-gap LL 1.431
- 2026 R05: post hit 0.0%, post LL 1.223, pole LL 34.539, quali-gap LL 1.924
- 2026 R06: post hit 100.0%, post LL 0.725, pole LL 0.000, quali-gap LL 1.585
- 2026 R07: post hit 0.0%, post LL 1.313, pole LL 34.539, quali-gap LL 1.734
- 2026 R08: post hit 0.0%, post LL 1.087, pole LL 0.000, quali-gap LL 1.652
- 2026 R09: post hit 0.0%, post LL 3.177, pole LL 34.539, quali-gap LL 1.688
- 2026 R10: post hit 0.0%, post LL 1.729, pole LL 0.000, quali-gap LL 1.310
- 2026 R11: post hit 100.0%, post LL 0.837, pole LL 0.000, quali-gap LL 1.577
- 2026 R12: post hit 100.0%, post LL 0.650, pole LL 0.000, quali-gap LL 1.716
- 2026 R13: post hit 0.0%, post LL 2.867, pole LL 34.539, quali-gap LL 2.546
- 2026 R14: post hit 100.0%, post LL 0.551, pole LL 34.539, quali-gap LL 1.771
- 2026 R15: post hit 100.0%, post LL 0.319, pole LL 0.000, quali-gap LL 0.616
- 2026 R16: post hit 100.0%, post LL 0.786, pole LL 0.000, quali-gap LL 1.278
- 2026 R17: post hit 100.0%, post LL 0.557, pole LL 0.000, quali-gap LL 1.662
- 2026 R18: post hit 0.0%, post LL 1.557, pole LL 0.000, quali-gap LL 1.575
- 2026 R19: post hit 0.0%, post LL 2.690, pole LL 34.539, quali-gap LL 2.257

## Transfer diagnostic (train <2026, test 2026 only)

- Train races: 92
- Test races (2026): 19
- Post-quali hit: **42.1%**
- Post-quali log-loss: **1.383**
- Quali-gap log-loss: **1.559**

## When model pick ≠ pole (train set)

- Races: 27
- Model hit: **40.7%**
- Log-loss: **1.193**


## All-season chronological out-of-sample window

- Test races: 91 (2022-11-13 to 2026-11-01)
- Every eligible race is held out once; each fit and calibration set contains earlier races only.
- Seasons: 2022, 2023, 2024, 2025, 2026
- Pre-quali races / hit / log-loss / prior-form log-loss: 91 / 40.7% / 1.670 / 2.403
- Pre-quali top-3 / Brier / rank correlation / finish MAE: 74.7% / 0.037 / 0.616 / 3.514
- Post-quali races / hit / log-loss / grid log-loss / quali-gap log-loss: 91 / 60.4% / 1.155 / 1.488 / 1.929
- Post-quali top-3 / Brier / rank correlation / finish MAE: 90.1% / 0.028 / 0.685 / 3.059
- DNF log-loss / Brier / PR-AUC / ROC-AUC: 0.385 / 0.113 / 0.297 / 0.579
- Post-sprint races / win hit / top-3 / log-loss / Brier: 12 / 25.0% / 66.7% / 1.524 / 0.033

- 2022: 2 races; pre LL 2.090; post LL 1.482; DNF PR-AUC 0.143
- 2023: 22 races; pre LL 1.084; post LL 0.769; DNF PR-AUC 0.259
- 2024: 24 races; pre LL 1.808; post LL 1.601; DNF PR-AUC 0.296
- 2025: 24 races; pre LL 1.813; post LL 1.005; DNF PR-AUC 0.261
- 2026: 19 races; pre LL 1.948; post LL 1.192; DNF PR-AUC 0.392

Training-set diagnostics are stored separately under `train_*` keys in `metrics.json`; they are not performance claims.
Win probability temperature is calibrated from preceding races in each fold; DNF probabilities are reported separately from conditional classified finishing distributions.

Important limitation: historical practice and weather snapshots are unavailable at forecast cutoffs, so current-event FP and weather features are excluded from both model feature views.
