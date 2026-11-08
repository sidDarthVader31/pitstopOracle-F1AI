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
- Pre-quali hit / top-3 / log-loss / Brier: **21.1% / 78.9% / 3.699 / 0.041**
- Prior-form baseline log-loss: **3.125**
- Post-quali hit / top-3 / log-loss / Brier: **42.1% / 68.4% / 1.548 / 0.033**
- Post-quali rank correlation / finish MAE: **0.750 / 2.989**
- Grid / quali-gap log-loss: **1.609 / 1.559**
- DNF log-loss / Brier / PR-AUC / ROC-AUC: **0.585 / 0.172 / 0.246 / 0.464**
- Post-sprint races / win hit / top-3 / log-loss / Brier: **6 / 16.7% / 83.3% / 1.888 / 0.046**

- 2026 R01: post hit 0.0%, post LL 2.417, pole LL 0.000, quali-gap LL 0.886
- 2026 R02: post hit 100.0%, post LL 0.717, pole LL 0.000, quali-gap LL 1.242
- 2026 R03: post hit 100.0%, post LL 0.587, pole LL 0.000, quali-gap LL 1.222
- 2026 R04: post hit 0.0%, post LL 1.370, pole LL 0.000, quali-gap LL 1.431
- 2026 R05: post hit 100.0%, post LL 0.324, pole LL 34.539, quali-gap LL 1.882
- 2026 R06: post hit 100.0%, post LL 0.670, pole LL 0.000, quali-gap LL 1.585
- 2026 R07: post hit 0.0%, post LL 1.424, pole LL 34.539, quali-gap LL 1.734
- 2026 R08: post hit 0.0%, post LL 2.053, pole LL 0.000, quali-gap LL 1.652
- 2026 R09: post hit 0.0%, post LL 2.185, pole LL 34.539, quali-gap LL 1.688
- 2026 R10: post hit 0.0%, post LL 1.312, pole LL 0.000, quali-gap LL 1.310
- 2026 R11: post hit 0.0%, post LL 1.870, pole LL 0.000, quali-gap LL 1.577
- 2026 R12: post hit 0.0%, post LL 2.223, pole LL 0.000, quali-gap LL 1.716
- 2026 R13: post hit 0.0%, post LL 3.352, pole LL 34.539, quali-gap LL 2.546
- 2026 R14: post hit 100.0%, post LL 0.832, pole LL 34.539, quali-gap LL 1.771
- 2026 R15: post hit 100.0%, post LL 0.636, pole LL 0.000, quali-gap LL 0.616
- 2026 R16: post hit 0.0%, post LL 1.081, pole LL 0.000, quali-gap LL 1.278
- 2026 R17: post hit 100.0%, post LL 0.939, pole LL 0.000, quali-gap LL 1.662
- 2026 R18: post hit 100.0%, post LL 0.128, pole LL 0.000, quali-gap LL 1.575
- 2026 R19: post hit 0.0%, post LL 5.298, pole LL 34.539, quali-gap LL 2.257

## Transfer diagnostic (train <2026, test 2026 only)

- Train races: 92
- Test races (2026): 19
- Post-quali hit: **52.6%**
- Post-quali log-loss: **1.443**
- Quali-gap log-loss: **1.559**

## When model pick ≠ pole (train set)

- Races: 41
- Model hit: **53.7%**
- Log-loss: **1.211**


## All-season chronological out-of-sample window

- Test races: 91 (2022-11-13 to 2026-11-01)
- Every eligible race is held out once; each fit and calibration set contains earlier races only.
- Seasons: 2022, 2023, 2024, 2025, 2026
- Pre-quali races / hit / log-loss / prior-form log-loss: 91 / 24.2% / 2.743 / 3.152
- Pre-quali top-3 / Brier / rank correlation / finish MAE: 65.9% / 0.042 / 0.638 / 3.489
- Post-quali races / hit / log-loss / grid log-loss / quali-gap log-loss: 91 / 58.2% / 1.729 / 1.466 / 1.954
- Post-quali top-3 / Brier / rank correlation / finish MAE: 79.1% / 0.030 / 0.720 / 3.028
- DNF log-loss / Brier / PR-AUC / ROC-AUC: 0.465 / 0.132 / 0.249 / 0.495
- Post-sprint races / win hit / top-3 / log-loss / Brier: 12 / 8.3% / 91.7% / 1.836 / 0.049

- 2022: 2 races; pre LL 2.465; post LL 0.761; DNF PR-AUC 0.218
- 2023: 22 races; pre LL 1.931; post LL 2.590; DNF PR-AUC 0.306
- 2024: 24 races; pre LL 3.923; post LL 1.889; DNF PR-AUC 0.230
- 2025: 24 races; pre LL 1.572; post LL 1.003; DNF PR-AUC 0.218
- 2026: 19 races; pre LL 3.699; post LL 1.548; DNF PR-AUC 0.246

Training-set diagnostics are stored separately under `train_*` keys in `metrics.json`; they are not performance claims.
Win probability temperature is calibrated from preceding races in each fold; DNF probabilities are reported separately from conditional classified finishing distributions.

Important limitation: historical practice and weather snapshots are unavailable at forecast cutoffs, so current-event FP and weather features are excluded from both model feature views.
