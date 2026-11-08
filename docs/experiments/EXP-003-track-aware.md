# EXP-003 — Track-aware features, race pace, monotonic pace model, race-weekend loop

**Status:** adopted as the weekend recipe (used for the 2026 Brazilian GP forecast); keep tracking live results
**Evaluation:** expanding walk-forward over the 43 races of 2025–2026. Each race is predicted only from earlier races, the same way it would have been predicted live.

## Plain-language summary

The model now knows what kind of track it's at, how each car does at that kind of track, and how fast each car has been in races (not only qualifying). Its probabilities no longer blow up to 0% for the eventual winner. It beats the "just trust the grid" baseline on average, and the track features are clearly the biggest improvement. It's still **not** reliably better than the previous model after qualifying across 2025–26. The difference is small and within the noise of 43 races.

## Questions

1. Does telling the model about track type and grid importance improve forecasts?
2. Can car strengths per track type be learned from results, without telemetry?
3. Which of the probability and data-pipeline fixes found in the earlier analysis matter?

## Changes

| Area | Change |
|---|---|
| Track profile | `ml/data/manual/circuit_profile.csv`: speed class (high/medium/low) and street flag for 27 circuits (manual, coarse). |
| Grid importance | `circuit_grid_importance`: per-race correlation between grid and finish, averaged over earlier races at the circuit and shrunk toward the all-circuit average. Monaco 0.77, Suzuka 0.74, Interlagos 0.52, Albert Park 0.48. Combined with grid and qualifying gap. |
| Car ratings | `car_pace_ewma` is now the qualifying gap to pole as a %. `car_quali_class_ewma` / `car_class_delta` give the team's pace at this track type compared with its overall pace. `car_race_pace_ewma` is the best race lap vs the field median from earlier races (OpenF1, 2023+). All reset in 2026. |
| Driver ratings | Teammate qualifying gap in % only. Race places vs teammate is a separate rating. Debutants start at a typical 0.3% behind their teammate instead of 0 ("always faster"). |
| Pace model | Monotonic constraints (`specs.MONOTONIC_DIRECTION`), e.g. a better grid slot or more points can never lower the predicted result. |
| DNF | The team's retirement rate in the current era, shrunk toward the field rate. Replaces the classifier, which had out-of-sample ROC-AUC of about 0.5. |
| Probabilities | Exact Plackett–Luce win probability averaged over DNF draws, so no 0% winners. Wider temperature grid. |
| Pipeline | The upcoming race is built by `build_driver_race`, the same code as training rows. Missing lap times are filled from qualifying position. Bugs fixed: teammate quali head-to-head was always 0; pit-stop averages were missing for the upcoming race. |
| Loop | `ml/race_weekend.py before/after`: retrain on all completed races, forecast, then score after the race into `reports/live_tracking.csv`. |

## Results (43 races, 2025–26)

| Variant | Post-quali win log-loss | Post-quali hit | Post-quali top-3 | Pre-quali log-loss | 2026 post-quali log-loss | 2026 post-quali hit |
|---|---:|---:|---:|---:|---:|---:|
| Previous model (`main`) | **1.244** | **55.8%** | 81.4% | 2.512 | 1.548 | **42.1%** |
| New fixes, no track features, no race pace | 1.444 | 41.9% | 83.7% | 1.930 | 1.654 | 26.3% |
| + track features | 1.352 | 39.5% | 88.4% | 1.973 | 1.578 | 26.3% |
| + track − monotonic constraints | 1.409 | 37.2% | 83.7% | 1.939 | 1.803 | 10.5% |
| **+ track + race pace (adopted)** | 1.322 | 41.9% | **88.4%** | 1.967 | **1.527** | 36.8% |
| adopted − track features | 1.478 | 37.2% | 86.0% | 1.937 | 1.704 | 15.8% |
| adopted + old blended car pace | 1.385 | 39.5% | 86.0% | 1.957 | 1.624 | 26.3% |
| adopted + old DNF classifier | 1.308 | 44.2% | 88.4% | 1.972 | 1.536 | 36.8% |
| Official grid baseline | 1.367 | 69.8% | — | — | 1.609 | 68.4% |
| Prior-form baseline (pre-quali) | — | — | — | 2.292 | — | — |

Log-loss measures how much probability the model gave the actual winner (lower is better). Hit rate is how often the model's top pick won.

- **Track features help most:** removing them worsens post-quali log-loss from 1.322 to 1.478, and the 2026 hit rate falls from 36.8% to 15.8%.
- **Monotonic constraints matter:** without them the 2026 log-loss jumps from 1.578 to 1.803.
- **Race pace helps.** The old seconds-based blended car rating hurts.
- **The old DNF classifier** is marginally better on 2025–26 post-quali log-loss but worse in 2026 and worse at predicting DNFs (log-loss 0.460 vs 0.415). We keep the transparent team rate.
- **Against the grid**, the adopted model has mean log-loss −0.045 (better), with a 95% bootstrap CI of [−0.53, +0.32]. It beats the grid in 24/43 races. Not yet statistically convincing.
- **Against the previous model**, it's worse after qualifying across 2025–26 (1.322 vs 1.244) but better in 2026, much better before qualifying (1.97 vs 2.51), and has no blow-up races (the previous model had one at log-loss 34.5).
- The grid "hit rate" of about 70% is still unmatched. The model spreads probability over the front-row cars, so its single top pick is often P2/P3. That costs hit rate but not log-loss.

### Full chronological window (91 races, 2022–2026, `ml/reports/EVAL.md`)

| | Previous model | Adopted |
|---|---:|---:|
| Post-quali win log-loss (grid baseline 1.466) | 1.729 | **1.362** |
| Post-quali hit / top-3 | 58.2% / 79.1% | 47.3% / **83.5%** |
| Pre-quali win log-loss (prior-form baseline: previous 3.152, now 2.386) | 2.743 | **1.795** |
| Pre-quali hit | 24.2% | **36.3%** |
| DNF ROC-AUC | 0.495 | **0.552** |

Over the full window the adopted model beats the grid baseline and the previous model on log-loss, both before and after qualifying.

### Grid-penalty follow-up

The Fantasy Lab what-if showed that moving the pole-sitter to P20 *raised* his win probability. The cause is `grid_vs_quali` (places lost to a penalty): in real data, big drops mostly hit fast cars taking engine penalties, so the model learned "big drop → strong car".

| 2025–26 post-quali | Log-loss | Hit | 2026 log-loss |
|---|---:|---:|---:|
| Adopted (unconstrained `grid_vs_quali`) | **1.322** | **41.9%** | **1.527** |
| `grid_vs_quali` constrained "never helps" | 1.388 | 39.5% | 1.621 |
| `grid_vs_quali` removed | 1.378 | 41.9% | 1.621 |

We keep the adopted recipe. The what-if now moves only the starting slot (`grid`, which is constrained so a worse slot never helps) and leaves `grid_vs_quali` unchanged.

## Decision

Adopt the "track + race pace" recipe as the weekend recipe and keep collecting live results. Revisit after 5 or more live races with `reports/live_tracking.csv`. Next experiments, one at a time:
1. Measured car traits from OpenF1 speed traps and sector times.
2. A ranking objective focused on the front of the field.
3. Weather scenarios once forecast snapshots accumulate.

## Limitations

- 43 races is a small sample, and the confidence intervals overlap with every baseline.
- Speed classes are manual and coarse. Race pace exists only from 2023.
- Weather isn't a trained input, because no historical forecasts are available. Brazil 2026 is forecast wet.
- Brazil 2026 qualifying lap times were missing from Jolpica, and OpenF1 times suggest wet qualifying. Gaps were filled from typical 2026 gaps by grid position.

## Reproduce

```bash
cd ml
F1ML_EVAL_FROM_SEASON=2025 python -c "from f1ml.train import train_and_eval; train_and_eval()"
# ablations: F1ML_TRACK_FEATURES=0 | F1ML_MONOTONIC=0 | F1ML_CAR_PACE=legacy | F1ML_DNF=hgb
```
