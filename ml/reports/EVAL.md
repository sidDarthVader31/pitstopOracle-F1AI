# F1 Oracle — evaluation (v2)

Time split: **train 2022–2026** (no held-out test; walk-forward below).
Separate **pre-quali** and **post-quali** models with race-level softmax calibration.

## Dataset

- Rows: 2102
- Races: 104
- Pole win rate: **59.6%**

## Walk-forward (by season)

- Season 2024: pre hit 29.2%, post hit 33.3%, post log-loss 2.658
- Season 2025: pre hit 25.0%, post hit 45.8%, post log-loss 2.338
- Season 2026: pre hit 41.7%, post hit 58.3%, post log-loss 2.172

Models: `ml/models/pre_quali/`, `ml/models/post_quali/`.
