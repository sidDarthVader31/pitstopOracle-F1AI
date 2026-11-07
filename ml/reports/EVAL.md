# F1 Oracle — evaluation (v3 ranker)

Training pool: **2022–2026** with era-reset constructor ratings and race-by-race walk-forward.
Primary shipping metric: **2026 walk-forward** log-loss vs pole and quali-gap softmax.

## Dataset

- Rows: 2256
- Races: 111
- Pole win rate: **59.5%**

## 2026 walk-forward (primary)

- Races scored: 19
- Post-quali hit rate (mean): **36.8%**
- Post-quali log-loss (mean): **1.581**
- Pole log-loss (mean): **10.907**
- Quali-gap log-loss (mean): **1.576**

- 2026 R01: post hit 100.0%, post LL 0.321, pole LL 0.000, quali-gap LL 0.968
- 2026 R02: post hit 0.0%, post LL 0.906, pole LL 0.000, quali-gap LL 1.436
- 2026 R03: post hit 100.0%, post LL 0.639, pole LL 0.000, quali-gap LL 1.222
- 2026 R04: post hit 0.0%, post LL 1.409, pole LL 0.000, quali-gap LL 1.431
- 2026 R05: post hit 100.0%, post LL 0.074, pole LL 34.539, quali-gap LL 1.924
- 2026 R06: post hit 100.0%, post LL 0.047, pole LL 0.000, quali-gap LL 1.585
- 2026 R07: post hit 100.0%, post LL 0.935, pole LL 34.539, quali-gap LL 1.734
- 2026 R08: post hit 100.0%, post LL 0.296, pole LL 0.000, quali-gap LL 1.652
- 2026 R09: post hit 0.0%, post LL 2.624, pole LL 34.539, quali-gap LL 1.688
- 2026 R10: post hit 0.0%, post LL 1.067, pole LL 0.000, quali-gap LL 1.310
- 2026 R11: post hit 0.0%, post LL 1.922, pole LL 0.000, quali-gap LL 1.577
- 2026 R12: post hit 0.0%, post LL 1.520, pole LL 0.000, quali-gap LL 1.716
- 2026 R13: post hit 0.0%, post LL 5.098, pole LL 34.539, quali-gap LL 2.546
- 2026 R14: post hit 0.0%, post LL 1.217, pole LL 34.539, quali-gap LL 1.771
- 2026 R15: post hit 0.0%, post LL 0.883, pole LL 0.000, quali-gap LL 0.616
- 2026 R16: post hit 0.0%, post LL 4.967, pole LL 0.000, quali-gap LL 1.278
- 2026 R17: post hit 0.0%, post LL 1.061, pole LL 0.000, quali-gap LL 1.662
- 2026 R18: post hit 100.0%, post LL 0.884, pole LL 0.000, quali-gap LL 1.575
- 2026 R19: post hit 0.0%, post LL 4.170, pole LL 34.539, quali-gap LL 2.257

## Transfer diagnostic (train <2026, test 2026 only)

- Train races: 92
- Test races (2026): 19
- Post-quali hit: **47.4%**
- Post-quali log-loss: **1.691**
- Quali-gap log-loss: **1.576**

## When model pick ≠ pole (train set)

- Races: 51
- Model hit: **54.9%**
- Log-loss: **1.022**


## Full walk-forward (by season)

- Season 2023: pre hit 0.0%, post hit 0.0%, post log-loss 1.663
- Season 2024: pre hit 100.0%, post hit 100.0%, post log-loss 0.636
- Season 2025: pre hit 0.0%, post hit 100.0%, post log-loss 0.656
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.321
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 0.906
- Season 2026: pre hit 0.0%, post hit 100.0%, post log-loss 0.639
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.409
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.074
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.047
- Season 2026: pre hit 0.0%, post hit 100.0%, post log-loss 0.935
- Season 2026: pre hit 0.0%, post hit 100.0%, post log-loss 0.296
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 2.624
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.067
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.922
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.520
- Season 2026: pre hit 100.0%, post hit 0.0%, post log-loss 5.098
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.217
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 0.883
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 4.967
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 1.061
- Season 2026: pre hit 100.0%, post hit 100.0%, post log-loss 0.884
- Season 2026: pre hit 0.0%, post hit 0.0%, post log-loss 4.170

Models: `ml/models/pre_quali/`, `ml/models/post_quali/`.
