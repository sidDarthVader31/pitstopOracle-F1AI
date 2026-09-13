# F1 Oracle — evaluation (v3 ranker)

Training pool: **2022–2026** with era-reset constructor ratings and race-by-race walk-forward.
Primary shipping metric: **2026 walk-forward** log-loss vs pole and quali-gap softmax.

## Dataset

- Rows: 2124
- Races: 105
- Pole win rate: **59.0%**

## 2026 walk-forward (primary)

- Races scored: 13
- Post-quali hit rate (mean): **46.2%**
- Post-quali log-loss (mean): **1.297**
- Pole log-loss (mean): **10.627**
- Quali-gap log-loss (mean): **1.599**

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

## Transfer diagnostic (train <2026, test 2026 only)

- Train races: 92
- Test races (2026): 13
- Post-quali hit: **38.5%**
- Post-quali log-loss: **1.482**
- Quali-gap log-loss: **1.599**

## When model pick ≠ pole (train set)

- Races: 45
- Model hit: **55.6%**
- Log-loss: **1.035**


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

Models: `ml/models/pre_quali/`, `ml/models/post_quali/`.
