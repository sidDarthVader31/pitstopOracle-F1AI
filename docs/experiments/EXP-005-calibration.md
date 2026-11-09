# EXP-005: Calibrate the favourites

**Status:** done. **Decision: keep** the "rolling temperature + grid blend" calibration as the recommended recipe. It is **not wired into the live weekend models**: changing the approved model is the owner's call (see Decision).
**Evaluation:** the same 91 walk-forward races as `ml/reports/EVAL.md` (2022 R21 to 2026 R19). 2025–26 (43 races) and 2026 (19 races) are shown separately. The internal market comparison uses the 50 races in EXP-004.
**Branch:** `claude/ml-fixes-6f0fa2`. No model artifact was written or promoted.

## Plain-language summary

The pace model ranks the drivers. A separate step, **calibration**, decides how *confident* the probabilities should be. When a forecast says "60%", that should happen about 6 times in 10. EXP-004 found the opposite for our strongest picks: post-quali favourites priced at 50%+ won only 39% of the time.

**Cause.** The confidence dial (the "temperature") was re-fitted for every race on just the 10 races before it. Ten races is a tiny sample. After a run of predictable races the dial jumped to "very sure". Early 2026 is the clearest case: the dial was fitted on late-2025 races, where the favourites kept winning, then applied to brand-new 2026 cars. At 2026 R9 the model gave its favourite 80%, and the eventual winner got 1.8%.

**Fix.** Fit the calibration on **every earlier walk-forward race**, not just the last 10. Also let it lean part of the way towards the plain "starting grid" forecast. Think of it as a race engineer who trusts the simulation but sanity-checks it against the grid. The calibration finds the right amount of grid on its own; it settles around 45–70% after qualifying.

**Result after qualifying:** 2025–26 log-loss drops from 1.410 to **1.265**, now better than the grid (1.386). In 2026 it drops from 1.637 to **1.513** (grid 1.617). Favourites at 50%+ now say 60% and win 65%. The model still loses clearly to the betting market (1.415 vs 1.172), so this changes nothing about positioning.

## Hypothesis

The over-confidence comes from the calibration step, not the pace model: a temperature fitted on 10 races is too noisy. A calibrator fitted on all earlier out-of-sample races will put the top calibration bucket within ±10 points and make post-quali log-loss beat the grid on 2025–26.

## The change

Only the step that turns pace scores into win probabilities changes. The pace model, DNF model and features are untouched. Every calibrator for race *k* is fitted only on walk-forward predictions for races *before k*: the same scores the model actually produced at the time, so nothing from race *k* or later is used. The first 10 walk-forward races keep the current calibration because there's no history yet.

| Variant | What it does |
|---|---|
| `current` | Today's recipe: one temperature fitted on the 10 races before the fold's training cut-off. |
| `rolling_T` | One temperature fitted on all earlier walk-forward races. |
| `rolling_T_last20` | Same, last 20 walk-forward races only. |
| `rolling_T_mix` | Temperature plus a mix with an equal-field forecast. |
| `rolling_T_floor1` | `rolling_T` with the temperature never below 1. |
| **`rolling_T_grid_blend`** | **Temperature plus a log-space blend with the grid baseline** (`eval.grid_probabilities`). Before qualifying there's no grid, so it blends with an equal field. Both numbers are fitted together. |
| `platt` | Logistic recalibration of the current win probability, then renormalised per race. |
| `isotonic` | Isotonic (monotone step-function) recalibration, then renormalised per race. |

Code: `ml/f1ml/calibration.py` (the blend, used by the experiment, with tests in `ml/tests/test_calibration.py`), `ml/scripts/exp_calibration.py` (all variants and scoring), and `ml/scripts/wf_dump.py` (caches walk-forward outputs so calibrators can be re-scored without refitting every fold).

**Log-loss** is the average "surprise" at the actual winner (lower is better). **Brier** is the mean squared error of every driver's probability. **"Top bucket"** is every driver-race priced at 50% or more: n, then what the model said on average, what happened, and a 95% interval for "what happened".

## Results: after qualifying (baseline = official grid)

Grid log-loss: 1.488 (all 91), **1.386 (2025–26)**, 1.617 (2026). Grid Brier 2025–26: 0.0289.

| Variant | All 91 | 2025–26 (model − grid, 95% CI) | 2026 | Brier 25–26 | Hit 25–26 | Top bucket, all 91 | Top bucket, 2025–26 | vs market (50 races; market 1.172) |
|---|---:|---|---:|---:|---:|---|---|---|
| current | 1.443 | 1.410 (+0.024, −0.40 to +0.39) | 1.637 | 0.0346 | 42% | 46: 69% / 59% (43–73%), **−11** | 23: 64% / 43%, **−21** | 1.485 (+0.31) |
| rolling_T | 1.446 | 1.370 (−0.016, −0.44 to +0.32) | 1.532 | 0.0329 | 40% | 36: 63% / 56%, −7 | 12: 58% / 42%, −16 | 1.460 (+0.29) |
| rolling_T_last20 | 1.454 | 1.367 (−0.020, −0.41 to +0.29) | 1.519 | 0.0329 | 42% | 37: 64% / 57%, −7 | 14: 57% / 43%, −14 | 1.455 (+0.28) |
| rolling_T_mix | 1.452 | 1.370 | 1.532 | 0.0329 | 40% | 36: 64% / 56%, −9 | 12: 58% / 42%, −16 | 1.460 (+0.29) |
| rolling_T_floor1 | 1.446 | 1.370 | 1.532 | 0.0329 | 40% | 36: 63% / 56%, −7 | 12: 58% / 42%, −16 | 1.460 (+0.29) |
| **rolling_T_grid_blend** | **1.402** | **1.265 (−0.122, −0.27 to +0.03)** | **1.513** | **0.0288** | **53%** | 37: 60% / 65% (47–80%), **+5** | 17: 55% / 65% (38–86%), **+9** | **1.415 (+0.24, CI +0.00 to +0.56)** |
| platt | 1.453 | 1.408 | 1.613 | 0.0345 | 42% | 39: 69% / 59%, −10 | 18: 64% / 39%, −25 | 1.484 (+0.31) |
| isotonic | 1.484 | 1.430 | 1.638 | 0.0347 | 42% | 40: 72% / 57%, −15 | 18: 67% / 39%, −28 | 1.517 (+0.35) |

Full calibration table for `rolling_T_grid_blend`, 2025–26 (n / said / happened):

| 0–2% | 2–5% | 5–10% | 10–20% | 20–35% | 35–50% | 50–100% |
|---|---|---|---|---|---|---|
| 663 / 0% / 0% | 52 / 4% / 2% | 52 / 7% / 2% | 46 / 15% / 9% | 42 / 28% / 38% | 20 / 42% / 50% | 17 / 55% / 65% |

On the market window (the same 50 races as EXP-004), the top bucket goes from 23 / 63% / 39% to **18 / 55% / 56%**. The market's own top bucket is 26 / 59% / 69%.

What we learn:
- **The 10-race temperature was the main problem.** Every variant that fits on the longer history fixes most of 2026 (1.637 → about 1.52–1.53).
- **The grid blend is what fixes the favourites and the Brier score.** Temperature alone still leaves 2025–26 favourites 16 points over-confident. Adding the grid gives the best log-loss, and Brier now matches the grid (0.0288 vs 0.0289; before it was 0.0346, much worse). The fitted grid weight stays between 0.45 and 0.7 from 2024 onwards (0.9 in mid-2023, when history was short).
- **Platt and isotonic don't help.** They recalibrate each driver's probability separately and then rescale per race, and that doesn't fix the favourites.
- The winner hit rate rises from 42% to 53% on 2025–26, still below the grid's 70%. The model keeps spreading probability across the front row.

## Results: before qualifying (baseline = prior form)

Prior-form log-loss: 2.403 (all), 2.311 (2025–26), 2.292 (2026).

| Variant | All 91 | 2025–26 (model − prior form, 95% CI) | 2026 | Top bucket, all 91 | Top bucket, 2025–26 | vs market before FP1 (48 races; market 1.735) |
|---|---:|---|---:|---|---|---|
| current | 1.803 | 1.981 (−0.330, −0.60 to −0.02) | 2.106 | 28: 72% / 64%, −7 | 6: 64% / 33%, −31 | 1.989 (+0.25) |
| rolling_T | 1.802 | 1.924 (−0.386, −0.64 to −0.11) | 2.139 | 32: 61% / 53%, −8 | 8: 59% / 25%, −34 | 1.949 (+0.21) |
| **rolling_T_grid_blend** (equal-field blend) | 1.810 | **1.903 (−0.407, −0.66 to −0.13)** | 2.129 | 29: 60% / 52%, −8 | 7: 58% / 43%, −15 | **1.937 (+0.20, CI −0.03 to +0.44)** |

Before qualifying the gain is smaller: 2025–26 improves, and 2026 is slightly worse (2.106 → 2.129). The 50%+ bucket holds only 6–8 driver-races in 2025–26, too few to judge.

## Done-criteria check (backlog P1)

| Criterion | Result |
|---|---|
| Post-quali top calibration bucket within ±10 points | **Met.** All 91 races: +5 (60% said, 65% won, n=37). 2025–26: +9 (n=17). Market window: +1 (n=18). Small buckets: the 95% interval for 2025–26 is 38–86%. |
| Post-quali log-loss beats the grid on 2025–26 walk-forward | **Met on the point estimate**: 1.265 vs 1.386, and 1.513 vs 1.617 in 2026. The 95% CI for the difference (−0.27 to +0.03) still just includes zero, so it isn't proven at 43 races. |

## Limitations

- **Chosen after looking.** Eight variants were compared on the same races and the best kept. The winner's margin is large compared with the alternatives, but some selection optimism is likely. The live 2027 record is the real test.
- **Half grid.** After qualifying, about half the final forecast now comes from the starting grid. That's honest (the grid is a strong signal), but the model's own contribution is smaller than the headline suggests. The model still beats the grid only when blended with it.
- **Not yet in production.** `race_weekend.py` still uses the 10-race temperature. Wiring this in needs a stored history of walk-forward predictions (`scripts/wf_dump.py` takes about 7 minutes) or a cheaper refit each weekend. Eval and live code must then share the same path (backlog P2, "align eval with production").
- The market gap remains (+0.24 after qualifying). This doesn't make the model a betting tool, and per the owner's decision it isn't meant to be one.
- Small samples: 43 races for 2025–26, and fewer than 20 driver-races in each top bucket.

## Decision

**Keep** `rolling_T_grid_blend` as the recommended calibration (post-quali and pre-quali). **Owner decision needed** to adopt it in the live weekend run, which means changing the approved model. Suggested order: (1) owner approves; (2) workstream D or E wires `f1ml.calibration` into `race_weekend.train_weekend_bundle` and the walk-forward, so `EVAL.md` reports the same path; (3) re-run `EVAL.md` and this experiment.

## Reproduce

```bash
cd ml
python scripts/wf_dump.py baseline          # ~7.5 min: walk-forward, caches per-driver outputs (git-ignored)
python scripts/exp_calibration.py baseline  # ~10 s: all variants + internal market check
# writes ml/reports/exp005_calibration_baseline.json
```

The market check needs `ml/data/cache/market/polymarket_snapshots.parquet` (from `scripts/market_benchmark.py fetch`). Market data is never a model input.
