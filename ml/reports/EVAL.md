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
- Pre-quali hit / top-3 / log-loss / Brier: **21.1% / 42.1% / 2.186 / 0.041**
- Prior-form baseline log-loss: **2.275**
- Post-quali hit / top-3 / log-loss / Brier: **36.8% / 78.9% / 1.527 / 0.036**
- Post-quali rank correlation / finish MAE: **0.702 / 3.099**
- Grid / quali-gap log-loss: **1.609 / 1.559**
- DNF log-loss / Brier / PR-AUC / ROC-AUC: **0.503 / 0.158 / 0.300 / 0.508**
- Post-sprint races / win hit / top-3 / log-loss / Brier: **6 / 16.7% / 83.3% / 1.470 / 0.037**

- 2026 R01: post hit 100.0%, post LL 1.295, pole LL 0.000, quali-gap LL 0.886
- 2026 R02: post hit 0.0%, post LL 1.955, pole LL 0.000, quali-gap LL 1.242
- 2026 R03: post hit 0.0%, post LL 1.166, pole LL 0.000, quali-gap LL 1.222
- 2026 R04: post hit 100.0%, post LL 0.517, pole LL 0.000, quali-gap LL 1.431
- 2026 R05: post hit 0.0%, post LL 1.034, pole LL 34.539, quali-gap LL 1.882
- 2026 R06: post hit 100.0%, post LL 0.511, pole LL 0.000, quali-gap LL 1.585
- 2026 R07: post hit 100.0%, post LL 0.494, pole LL 34.539, quali-gap LL 1.734
- 2026 R08: post hit 0.0%, post LL 2.816, pole LL 0.000, quali-gap LL 1.652
- 2026 R09: post hit 0.0%, post LL 3.466, pole LL 34.539, quali-gap LL 1.688
- 2026 R10: post hit 0.0%, post LL 1.898, pole LL 0.000, quali-gap LL 1.310
- 2026 R11: post hit 0.0%, post LL 1.971, pole LL 0.000, quali-gap LL 1.577
- 2026 R12: post hit 0.0%, post LL 1.613, pole LL 0.000, quali-gap LL 1.716
- 2026 R13: post hit 0.0%, post LL 2.412, pole LL 34.539, quali-gap LL 2.546
- 2026 R14: post hit 0.0%, post LL 1.399, pole LL 34.539, quali-gap LL 1.771
- 2026 R15: post hit 100.0%, post LL 0.574, pole LL 0.000, quali-gap LL 0.616
- 2026 R16: post hit 100.0%, post LL 0.555, pole LL 0.000, quali-gap LL 1.278
- 2026 R17: post hit 100.0%, post LL 1.038, pole LL 0.000, quali-gap LL 1.662
- 2026 R18: post hit 0.0%, post LL 1.268, pole LL 0.000, quali-gap LL 1.575
- 2026 R19: post hit 0.0%, post LL 3.037, pole LL 34.539, quali-gap LL 2.257

## Transfer diagnostic (train <2026, test 2026 only)

- Train races: 92
- Test races (2026): 19
- Post-quali hit: **36.8%**
- Post-quali log-loss: **1.445**
- Quali-gap log-loss: **1.559**

## When model pick ≠ pole (train set)

- Races: 40
- Model hit: **45.0%**
- Log-loss: **1.154**


## All-season chronological out-of-sample window

- Test races: 91 (2022-11-13 to 2026-11-01)
- Every eligible race is held out once; each fit and calibration set contains earlier races only.
- Seasons: 2022, 2023, 2024, 2025, 2026
- Pre-quali races / hit / log-loss / prior-form log-loss: 91 / 36.3% / 1.795 / 2.386
- Pre-quali top-3 / Brier / rank correlation / finish MAE: 69.2% / 0.037 / 0.615 / 3.498
- Post-quali races / hit / log-loss / grid log-loss / quali-gap log-loss: 91 / 47.3% / 1.362 / 1.466 / 1.911
- Post-quali top-3 / Brier / rank correlation / finish MAE: 83.5% / 0.032 / 0.678 / 3.097
- DNF log-loss / Brier / PR-AUC / ROC-AUC: 0.391 / 0.115 / 0.280 / 0.552
- Post-sprint races / win hit / top-3 / log-loss / Brier: 12 / 25.0% / 91.7% / 1.492 / 0.038

- 2022: 2 races; pre LL 2.003; post LL 1.373; DNF PR-AUC 0.157
- 2023: 22 races; pre LL 1.214; post LL 1.114; DNF PR-AUC 0.261
- 2024: 24 races; pre LL 2.001; post LL 1.659; DNF PR-AUC 0.316
- 2025: 24 races; pre LL 1.794; post LL 1.159; DNF PR-AUC 0.257
- 2026: 19 races; pre LL 2.186; post LL 1.527; DNF PR-AUC 0.300

Training-set diagnostics are stored separately under `train_*` keys in `metrics.json`; they are not performance claims.
Win probability temperature is calibrated from preceding races in each fold; DNF probabilities are reported separately from conditional classified finishing distributions.

Important limitation: historical practice and weather snapshots are unavailable at forecast cutoffs, so current-event FP and weather features are excluded from both model feature views.
