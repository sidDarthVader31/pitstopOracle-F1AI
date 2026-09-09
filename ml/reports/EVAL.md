# F1 Oracle — evaluation (v3 ranker)

Training pool: **2022–2026** with era-reset constructor ratings and race-by-race walk-forward.
Primary shipping metric: **2026 walk-forward** log-loss vs pole and quali-gap softmax.

## Dataset

- Rows: 2102
- Races: 104
- Pole win rate: **59.6%**

## 2026 walk-forward (primary)

- Races scored: 12
- Post-quali hit rate (mean): **66.7%**
- Post-quali log-loss (mean): **0.764**
- Pole log-loss (mean): **8.635**
- Quali-gap log-loss (mean): **1.520**

- 2026 R01: post hit 100.0%, post LL 0.660, pole LL 0.000, quali-gap LL 0.968
- 2026 R02: post hit 0.0%, post LL 0.823, pole LL 0.000, quali-gap LL 1.436
- 2026 R03: post hit 100.0%, post LL 0.307, pole LL 0.000, quali-gap LL 1.222
- 2026 R04: post hit 0.0%, post LL 1.515, pole LL 0.000, quali-gap LL 1.431
- 2026 R05: post hit 100.0%, post LL 0.058, pole LL 34.539, quali-gap LL 1.924
- 2026 R06: post hit 100.0%, post LL 0.009, pole LL 0.000, quali-gap LL 1.585
- 2026 R07: post hit 0.0%, post LL 1.951, pole LL 34.539, quali-gap LL 1.734
- 2026 R08: post hit 100.0%, post LL 0.627, pole LL 0.000, quali-gap LL 1.652
- 2026 R09: post hit 100.0%, post LL 1.315, pole LL 34.539, quali-gap LL 1.688
- 2026 R10: post hit 100.0%, post LL 0.521, pole LL 0.000, quali-gap LL 1.310
- 2026 R11: post hit 100.0%, post LL 0.194, pole LL 0.000, quali-gap LL 1.577
- 2026 R12: post hit 0.0%, post LL 1.187, pole LL 0.000, quali-gap LL 1.716

## Transfer diagnostic (train <2026, test 2026 only)

- Train races: 92
- Test races (2026): 12
- Post-quali hit: **33.3%**
- Post-quali log-loss: **1.412**
- Quali-gap log-loss: **1.520**

## When model pick ≠ pole (train set)

- Races: 44
- Model hit: **61.4%**
- Log-loss: **0.964**


## Full walk-forward (by season)

- Season 2023: pre hit 0.0%, post hit 0.0%, post log-loss 1.663
- Season 2024: pre hit 0.0%, post hit 100.0%, post log-loss 0.374
- Season 2025: pre hit 0.0%, post hit 100.0%, post log-loss 0.570
- Season 2026: pre hit 0.0%, post hit 100.0%, post log-loss 0.660
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 0.823
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.307
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.515
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.058
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.009
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.951
- Season 2026: pre hit 0.0%, post hit 100.0%, post log-loss 0.627
- Season 2026: pre hit 0.0%, post hit 100.0%, post log-loss 1.315
- Season 2026: pre hit 0.0%, post hit 100.0%, post log-loss 0.521
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.194
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.187

Models: `ml/models/pre_quali/`, `ml/models/post_quali/`.
