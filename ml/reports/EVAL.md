# F1 winner model — evaluation

Time split: **train 2022–2024**, **test 2025–2026** (held-out later seasons).

## Dataset

- Rows (driver-race): 2102
- Races: 104 (train 68, test 36)
- Pole sitter win rate (all data): **59.6%**
- Wet races (daily precipitation ≥ 1 mm): 39.4%
- Overall DNF rate: 33.1%

See `pole_win_rate.png` and `dnf_by_constructor.png`.

## Winner hit rate (share of races where predicted winner is correct)

| Method | Train | Test |
|---|---:|---:|
| Baseline: pole | 54.4% | 69.4% |
| Baseline: championship leader | 60.3% | 30.6% |
| Logistic regression | 57.4% | 52.8% |
| Random forest | 83.8% | 58.3% |

## Top-3 (actual winner in model's top 3 probabilities)

- Train RF: 100.0%
- Test RF: 100.0%
- Train logreg: 89.7%
- Test logreg: 86.1%

## Did ML beat pole?

Not on this split — test RF 58.3% vs pole 69.4%. That is expected with ~20 races/year, DNFs, and qualifying already explaining most winners.

Saved models: `ml/models/winner_rf.joblib`, `ml/models/winner_logreg.joblib`.
