# ML launch readiness: integrity re-audit (workstream B)

**Date:** 2026-11-09. **Branch:** `claude/market-benchmark-25777b`, based on 75ffc3f (official starting grid).
**Scope:** re-check eight suspected integrity risks against the **current** code, fix confirmed leakage, and judge which model claims are safe to make at launch. Market comparison: [`experiments/EXP-004-market-benchmark.md`](experiments/EXP-004-market-benchmark.md).

**"Leakage"** means the model or its evaluation using information that wouldn't exist at forecast time. In F1 terms, it's like "predicting" the race winner on Saturday night using Sunday's lap chart. It makes the scores look better than they really are.

## Summary

| # | Risk | Result | One-line finding |
|---|---|---|---|
| 1 | Current-race `race_pace_delta` leaking into pre-race features | **PASS** | It isn't a model feature. `car_race_pace_ewma` is assigned before the race's own pace is folded in. A new regression test guards this. |
| 2 | Observed weather in training vs forecast weather live | **PASS** | No weather column is a model feature. Wet weather is only an explicit what-if. |
| 3 | Practice-session eligibility on sprint weekends | **PASS (not applicable)** | No practice feature is used by any model stage. |
| 4 | `sprint_position` used before the sprint | **PASS for leakage / FIXED live stage selection** | The feature only enters `post_sprint`. The live code used to switch to `post_sprint` as soon as a 2024+ sprint finished, before Grand Prix qualifying existed. Fixed by workstream D on 2026-11-09 (see item 4). |
| 5 | DNF training population filtered to classified finishers | **PASS** | Retirements are kept. Only DNS, DSQ and withdrawals are unlabelled. |
| 6 | Fixed DNF score penalty | **PASS** | Gone. DNF is simulated separately in the Monte Carlo. Related gap: the pace model also learns from retirees' back-of-field positions. |
| 7 | Eval vs inference mismatches; mislabelled `test_*` metrics | **FAIL → FIXED (leak), PARTIAL (rest)** | **Leak fixed:** the walk-forward scored only drivers who later got a DNF label, dropping disqualified and non-starting drivers with hindsight. Smaller mismatches remain (listed below). The `test_*` keys are out-of-sample but cover 2026 only. |
| 8 | 2026 regulation-era handling | **PARTIAL** | Car ratings reset, recency weights and the era DNF rate are in place. The model still doesn't beat the grid in 2026 (1.637 vs 1.617), and the era year is hard-coded in three places. |

`pytest ml/tests`: **51 passed, 1 skipped** (3 new regression tests).

## Evidence

### 1. `race_pace_delta` → pre-race features: PASS
- `race_pace_delta` and `race_pace_pct` are built from the race session itself (`ml/f1ml/features.py:71`, `ml/f1ml/features.py:86`), so they are post-race facts.
- Neither is in any model allowlist (`ml/f1ml/specs.py:17-75`). `ml/tests/test_features.py:59` asserts `race_pace_delta` is excluded from all three stages.
- The only model input derived from them is `car_race_pace_ewma`. In `build_ratings`, each event's rating is written **before** that event's race pace updates the running state (`ml/f1ml/ratings.py:173` assigns, `ml/f1ml/ratings.py:202` updates afterwards). It's a shifted average.
- `car_pace_legacy_ewma` also uses `race_pace_delta`, with the same assign-then-update order. It's only used with `F1ML_CAR_PACE=legacy`.
- **New test:** `ml/tests/test_ranking.py::test_current_race_outcomes_cannot_move_pre_race_ratings` scrambles a race's own race pace, qualifying, result and DNF. The ratings entering that race stay identical.

### 2. Observed vs forecast weather: PASS
- Observed weather (Open-Meteo archive) is merged into every row (`ml/f1ml/features.py:474`). Forecast weather overwrites it only for the upcoming race (`ml/f1ml/features.py:159-188`).
- `is_wet`, `precipitation_mm`, `rain_mm` and `weathercode` are **not** in any feature list (`ml/f1ml/specs.py:17-75`). This is asserted at `ml/tests/test_features.py:59`. So training never sees observed weather, and live scoring never sees forecast weather.
- Wet weather enters only as the explicit, user-chosen what-if (`ml/f1ml/predict.py:546-562`).
- The legacy DNF classifier (`F1ML_DNF=hgb`) lists `is_wet` and `precipitation_mm` as candidates, but filters them against the allowlist when training (`ml/f1ml/ranking.py:180-221`) and imputes them when scoring (`ml/f1ml/ranking.py:248`).
- Remaining gap, not a leak: the model has **no weather signal at all**. Wet races are its weakest. See backlog P2.

### 3. Practice eligibility on sprint weekends: PASS (not applicable)
- FP1–FP3 deltas are computed (`ml/f1ml/features.py:51-126`) but **no stage uses them** (`ml/f1ml/specs.py:1-5`, `ml/tests/test_features.py:59`). So the "only FP1 exists on sprint weekends" eligibility problem can't affect current forecasts.
- If practice features are reintroduced (backlog P2), eligibility per weekend format must be solved first. On 2024+ sprint weekends only FP1 happens before Sprint Qualifying. In 2023, FP2 ran after GP qualifying.

### 4. `sprint_position` before the sprint: PASS (leakage) / FAIL (live stage selection)
- `sprint_position` is only added in `post_sprint` (`ml/f1ml/specs.py:60`, `ml/f1ml/specs.py:148`). It's excluded from `pre_quali` and `post_quali` (asserted at `ml/tests/test_features.py:59`). The walk-forward only scores `post_sprint` on rows where the sprint has a result (`ml/f1ml/train.py`, the `te_sprint` block).
- **Mismatch (not fixed: it's an eval/inference mismatch, not leakage).** `race_weekend._stage_for` (`ml/race_weekend.py:48-54`) and `predict._race_status_row` / `race_status` (`ml/f1ml/predict.py:136-154`) choose `post_sprint` whenever a sprint result exists. On the 2024+ sprint format the sprint runs **before** Grand Prix qualifying. So between Saturday's sprint and qualifying, the live forecast uses the `post_sprint` model with **no qualifying or grid inputs**, which get imputed. That model was trained only on rows that had both. Fix: choose `post_sprint` only when qualifying is also known; before that, use `pre_quali`, which ignores the sprint, or train a "post-sprint, pre-quali" stage. Backlog **P1**.
- **Fixed (workstream D, 2026-11-09, commit "Fix sprint-weekend stage selection").** `_stage_for`, `_race_status_row` and `race_status` now need GP qualifying as well as a sprint result before choosing `post_sprint`. Between the sprint and qualifying they use `pre_quali`. Regression tests: `ml/tests/test_weekend_predict.py::test_weekend_stage_needs_quali_before_post_sprint` and `::test_race_status_needs_quali_before_post_sprint`.

### 5. DNF training population: PASS
- `_load_training_frame` keeps every row with a DNF label (`ml/f1ml/train.py:44-48`). Retirements are labelled 1 and kept: 311 DNF rows vs 1,916 finishers.
- `_dnf_labels` (`ml/f1ml/features.py:23-28`) leaves only "Did not start", "Disqualified" and "Withdrew" unlabelled. Correct: those aren't reliability failures.
- `EmpiricalDNFModel.fit` uses all labelled rows (`ml/f1ml/ranking.py:39-54`). Tests: `ml/tests/test_train_eval.py::test_training_frame_keeps_non_position_dnf_rows`, `ml/tests/test_ranking.py:90`.

### 6. Fixed DNF score penalty: PASS
- No fixed penalty remains in the model path. Each Monte Carlo draw retires drivers with their own DNF probability, then ranks only the classified drivers (`ml/f1ml/ranking.py:269-301`). `DNF_PENALTY = -15` in `ml/f1ml/fantasy.py:33` is the F1 Fantasy scoring rule, not a model adjustment. Check its value against the 2027 Fantasy rules before launch.
- **Related gap (backlog P2):** the pace model trains on *every* numeric finishing position, including retirees listed at the back (`ml/f1ml/ranking.py:153`; the mean position of DNF rows is 18.8). Unreliable cars are therefore pushed down twice: once in "pace" and once by the DNF simulation.

### 7. Eval vs inference and `test_*` labels: FAIL → FIXED (leak), PARTIAL (rest)

**Confirmed leak, now fixed.** The walk-forward used `_load_training_frame` for test races too. That frame drops any row without a DNF label, so **drivers later disqualified, or who didn't start, were removed from the field being scored**. A real forecast can't know that in advance. For example, at 2024 Spa, George Russell was disqualified after finishing first on the road. Removing him hands his win probability to the rest of the field, Hamilton included, using hindsight. 29 driver-races in completed races were affected.
- Fix: a new `_load_eval_field()` (`ml/f1ml/train.py:51-63`) returns every entrant of every completed race. `_walk_forward_folds(..., test_field=...)` uses it for the test race only, so training is unchanged. DNF metrics use only labelled rows (`ml/f1ml/train.py:534`).
- Regression tests: `ml/tests/test_train_eval.py::test_eval_field_keeps_disqualified_and_non_starters` and `::test_walk_forward_test_fold_uses_full_entry_list`.
- Re-run (`ml/reports/EVAL.md`, 91 races): post-quali log-loss **1.422 → 1.443**, grid baseline 1.466 → 1.488, so the gap over the grid is unchanged at about 0.045. Pre-quali 1.788 → 1.803 (prior form 2.386 → 2.403). 2026 post-quali **1.618 → 1.637** (grid 1.609 → 1.617). Post-quali hit rate 48.4% → 47.3%. The old numbers were slightly flattering.

**Remaining mismatches (not leaks):**

| Mismatch | Eval | Live | Impact |
|---|---|---|---|
| Final fit | **Fixed 2026-11-10:** both call `train.train_stage_bundle` (train on all earlier races; temperature from earlier out-of-sample forecasts) | Same function | None for pre/post-quali. post_sprint still calibrates on a model that is then refit, in both paths. |
| Monte Carlo samples | 1,200 per race (`ml/f1ml/train.py:132`); temperature fit uses 100 (`ml/f1ml/train.py:196`) | 6,000 (`ml/race_weekend.py:38`); temperature fit uses 300 | Win probabilities use the exact formula, so the impact is mainly on podium and points noise. |
| Sprint-weekend stage | `post_sprint` evaluated only with qualifying known | **Fixed:** now also needs qualifying (item 4) | None remaining. |

**`test_*` metric labels.** `metrics.json` `test_pre_quali`, `test_post_quali`, `test_rf_hit` and `test_rf_top3` are genuinely out-of-sample. But they average **2026 walk-forward races only** (`ml/f1ml/train.py:617`, `ml/f1ml/train.py:628-639`), while the dashboard reads them as generic "test" performance (`ml/f1ml/ui/charts.py:55-86`). "rf" is a leftover from an old random-forest model. Not a leak: rename these in the UI to "2026 out-of-sample" when the dashboard is reworked (backlog P3).

### 8. 2026 regulation era: PARTIAL
- In place:
  - Car ratings reset at the era change (`ml/f1ml/ratings.py:165-171`).
  - `regs_era` feature (`ml/f1ml/ratings.py:20`, `ml/f1ml/ratings.py:76`).
  - Recency weights: the test season ×6, previous ×2.5 (`ml/f1ml/ranking.py:128-136`).
  - DNF rate from the current era once 5 or more races exist (`ml/f1ml/ranking.py:43-47`).
  - Quali-gap imputation only uses the same era (`ml/f1ml/features.py:298-328`).
- Gaps:
  - The model doesn't beat the grid in 2026: post-quali 1.637 vs grid 1.617 and quali-gap 1.576 (19 races). Against the market it is 1.637 vs 1.165 (EXP-004).
  - The era year is defined twice (`ml/f1ml/specs.py:14`, `ml/f1ml/ratings.py:8`) and hard-coded as `2026` at `ml/f1ml/features.py:308`.
  - Driver ratings deliberately carry over, which is untested.
  - For 2027 (same regulations), check that the recency weighting keeps 2026 dominant and doesn't fall back to 2022–2025 patterns.

## Can the DNF model support launch claims? No.

- Out of sample, 91 races: **ROC-AUC 0.552**, PR-AUC 0.280 against a base rate of about 14%. In 2026: **ROC-AUC 0.508**. ROC-AUC 0.5 is a coin flip. In F1 terms, it ranks "who will retire" barely better than drawing names from a helmet.
- The production DNF model (`EmpiricalDNFModel`, `ml/f1ml/ranking.py:27-58`) is just each team's retirement rate this era, shrunk toward the field average. That's honest and transparent. It's *useful* for spreading win and podium probability realistically. It is *not* a predictive edge.
- **Allowed at launch:** "DNF % = this team's retirement rate this season (shrunk to the field average)", shown as context.
- **Not allowed:** "we predict retirements", DNF-based betting angles, or any accuracy claim about DNFs.

## What claims are safe at launch (as of this audit)

| Claim | OK? | Basis |
|---|---|---|
| "Forecasts are tested on races the model hadn't seen" | Yes | Walk-forward, now with the full entry list |
| "Beats the starting grid after qualifying" | **Only with caveats** | 1.443 vs 1.488 over 91 races, inside the noise (EXP-003 bootstrap). Loses in 2026 (1.637 vs 1.617). |
| "Better than prior form before qualifying" | Yes | 1.803 vs 2.403, consistent across seasons |
| "Sharper than the odds" / "value bets" | **No** | Loses to the market by 0.25–0.31 log-loss (EXP-004) |
| "Predicts DNFs" | **No** | ROC-AUC 0.55 (0.51 in 2026) |
| "Well-calibrated" | **No** | Post-quali favourites at 50%+ win 39% of the time (EXP-004) |

## Prioritized backlog for the ML-improvement workstream (D)

| Priority | Item | Why | Done when |
|---|---|---|---|
| **P1** | Calibrate the top of the grid. Favourites at 50%+ win 39% vs 63% predicted. Try a separate temperature, or isotonic calibration on the favourite's probability. | Biggest single loss against both grid and market. A G3 requirement ("probabilities are calibrated"). | The calibration table's top bucket is within ±10 points. Post-quali log-loss beats the grid in 2025–26. |
| **P1** | Fix sprint-weekend stage selection (item 4). | The live forecast between the sprint and GP qualifying uses a model without its key inputs. | `post_sprint` only once qualifying is known. Test added. |
| **P1** | Make the market a standing benchmark. Save Polymarket (and, if approved, Betfair) snapshots with every live forecast, and re-run `scripts/market_benchmark.py` each race. | G1 and G3 need a published live record against the market. | Snapshots saved by `race_weekend.py before`. |
| **P1** | Try "market + model" forecasts: market as a feature, or as a prior the model adjusts. | The only realistic path to something sharper than the market. The 50/50 blend already cuts model log-loss from 1.49 to 1.24. | Beats the de-margined market on walk-forward, CI excluding zero. |
| P2 | Align eval with production: refit on all races after temperature calibration, and use the same Monte Carlo sample counts. | Removes the remaining eval/inference mismatch (item 7). | The eval path calls the same function as `race_weekend.train_weekend_bundle`. |
| P2 | Stop double-counting retirements: train the pace model on classified finishers, or treat DNF positions as censored. | Item 6 gap. | Ablation recorded as an EXP. |
| P2 | Weather: save forecast snapshots each weekend (started in `data/canonical/weather_forecast.parquet`). Once 10 or more exist, test a wet-race adjustment. | The model has no weather signal. | EXP with wet vs dry split. |
| P2 | Practice pace with per-format eligibility rules (item 3). | The market gains ~0.06 log-loss from practice (pre-weekend 1.735 → pre-quali 1.675). The model has none. | Feature ablation EXP. Eligibility table per weekend format. |
| P3 | One `ERA_RESET_YEAR` constant (replace the copy in `ratings.py` and the literal at `features.py:308`). | Hygiene before the next regulation change. | grep finds one definition. |
| P3 | Rename the `test_*` / `test_rf_*` metric keys and their UI labels to "2026 out-of-sample". | Avoids misleading accuracy text. | Dashboard shows the correct label. |
| P3 | Replace or retire the DNF model's predictive role (survival model, part failures, penalties). | ROC-AUC ~0.5. | Out-of-sample ROC-AUC > 0.6 before any DNF claim. |

## Workstream D results and G3 status (2026-11-09)

**Branch:** `claude/ml-fixes-6f0fa2`. One commit per fix or experiment. `pytest ml/tests`: **66 passed, 1 skipped**. No model artifact was written or promoted. The live weekend recipe is unchanged until the owner approves.

| Backlog item | Outcome | Evidence |
|---|---|---|
| P1 Sprint-weekend stage selection | **Fixed** | Item 4 above. Regression tests added. |
| P1 Calibrate the favourites | **Done: keep** a calibration fitted on all earlier walk-forward races. On today's pace model, rolling temperature + grid blend: 2025–26 post-quali 1.265 vs grid 1.386, top bucket 60% said / 65% won. | [`EXP-005`](experiments/EXP-005-calibration.md) |
| P2 Stop double-counting retirements | **Done: keep** `F1ML_PACE_TRAIN=classified`. Post-quali 1.443 → 1.305 (all 91), 1.410 → 1.225 (2025–26), 1.637 → 1.249 (2026; the CI excludes zero). Favourites stop being over-confident. Post-sprint stage worse (12 races). | [`EXP-006`](experiments/EXP-006-classified-pace.md) |
| Fantasy outputs check | **Done.** Expected finish is better than baselines (0.2–0.45 places). Pre-quali podium is slightly better. Points-finish chances and pick-5 are no better than grid- or form-rank tables. `fantasy.py` expected points fixed to use the simulated distribution (live dashboard). | [`EXP-007`](experiments/EXP-007-fantasy-outputs.md) |
| P3 One `ERA_RESET_YEAR` | **Done.** Only `specs.py` defines it. | Hygiene commit |
| P3 `test_*` keys and UI labels | **Done.** Keys are now `oos2026_*` and `train_model_*`. The dashboard says "2026 out-of-sample" and still reads old `metrics.json` files. | Hygiene commit, `tests/test_dashboard_contract.py` |

**Recommended recipe** (owner approval needed): pace model on classified finishers (EXP-006) + temperature fitted on all earlier walk-forward races (EXP-005). Walk-forward post-quali log-loss: **1.276** over all 91 races (grid 1.488), **1.184** in 2025–26 (grid 1.386; difference −0.20, 95% CI −0.62 to +0.10), **1.210** in 2026 (grid 1.617). The 2025–26 top bucket is 59% said / 59% won. Internal market benchmark: 1.289 vs market 1.172 (+0.12, CI −0.02 to +0.26). Still behind the market, and that's only a benchmark.

### G3 "model launch-worthy" (target 2027-01-31)

| G3 condition | Approved model today | Recommended recipe (EXP-005 + EXP-006) | Status |
|---|---|---|---|
| No known leakage | No leak open after workstream B's audit. D's changes affect training rows and calibration only, both fitted on earlier races. | Same | **Pass** |
| Post-quali beats the grid on walk-forward | **No** for 2025–26 (1.410 vs 1.386) and 2026 (1.637 vs 1.617). Yes over all 91 (1.443 vs 1.488). | **Yes on point estimates** in every window (1.184 vs 1.386 in 2025–26; 1.210 vs 1.617 in 2026). Not statistically proven: the CI includes zero. The grid-blend calibration variant's 2025–26 CI excludes zero (−0.41 to −0.02). | **Met only if the recipe is adopted** |
| Probabilities are calibrated | **No.** Winner top bucket 64% said / 43% won in 2025–26. | **Winner: yes.** Top bucket within ±10 points in every window (n = 17–46, wide intervals). **Points-finish (top 10): no.** Midfield over-confident (EXP-007). | **Partial** |
| Live shadow record published | Not workstream D's task (coordinator) | — | **Open** |

**Overall G3 verdict: not passed yet, but on track.** What's left, in order:
1. The owner approves the recommended recipe (it changes the approved model).
2. Wire it into `race_weekend.train_weekend_bundle` **and** the walk-forward through one shared function (backlog P2, "align eval with production"). Keep `post_sprint` on the old training set or re-check it. Regenerate `EVAL.md` and re-run `scripts/market_benchmark.py report`.
3. Calibrate points-finish probabilities: a separate temperature for positions beyond the winner (EXP-007 follow-up). Needed before the product shows "chance of points" as a calibrated number.
4. The coordinator's live shadow record.

### Claims after workstream D

| Claim | OK? | Basis |
|---|---|---|
| "Beats the starting grid after qualifying" | Only once the recipe is adopted, and with a caveat ("on average, over 43 races") | EXP-005/006 |
| "Well-calibrated win chances" | Only once the recipe is adopted | EXP-005/006 |
| "Expected finishing position is more accurate than grid/form" | **Yes** | EXP-007: CI excludes zero in every window |
| "Calibrated chance of scoring points" | **No** | EXP-007 |
| "Helps you pick a better Fantasy team" | **No** (same as top-5 by grid or form) | EXP-007 |

### Update 2026-11-10: recipe adopted (owner decision)

The owner approved the recommended recipe. Code changes:
- `ranking.default_pace_train`: classified finishers for pre_quali and post_quali; post_sprint unchanged.
- `train.train_stage_bundle`: the single recipe used by `race_weekend.train_weekend_bundle` and the walk-forward.
- `reports/oos_calibration_history.parquet`: written by the walk-forward and extended by `race_weekend.py after`.
- Tests: `ml/tests/test_adopted_recipe.py`, including "each fold calibrates only on earlier forecasts". `pytest ml/tests`: 74 passed, 1 skipped.

Regenerated `ml/reports/EVAL.md` and `MARKET_BENCHMARK.md`:

| | Before workstream D | Adopted |
|---|---:|---:|
| Post-quali log-loss, all 91 (grid 1.488) | 1.443 | **1.155** (−0.33 vs grid, CI −0.58 to −0.11) |
| Post-quali, 2025–26 (grid 1.386) | 1.410 | **1.088** (−0.30, CI −0.71 to +0.01) |
| Post-quali, 2026 (grid 1.617) | 1.637 | **1.192** |
| Post-quali hit / Brier (all) | 47.3% / 0.033 | 60.4% / 0.028 (grid 0.031) |
| Top bucket 50%+, 2025–26 (said / won) | 64% / 43% | **60% / 66%** (n=29) |
| Pre-quali log-loss, all (prior form 2.403) | 1.803 | **1.670** |
| Pre-quali top bucket, all | 72% / 64% | 69% / 59% (n=27). Small-sample 2025–26 favourites still over-confident. |
| Post-sprint log-loss (12 races) | 1.369 | 1.524 (worse; the eval now refits like production) |
| DNF ROC-AUC, all / 2026 | 0.55 / 0.51 | 0.58 / 0.63 |
| Internal market benchmark, post-quali (market 1.172) | 1.485 | 1.250 (+0.08, CI −0.10 to +0.28; better in 25 of 50) |

**Why it's better than the EXP-005/006 estimate (1.276).** The walk-forward used to hold out the 10 most recent races, the most heavily weighted ones, only to set the temperature. Now the temperature comes from earlier forecasts, so the model trains on those races too, exactly as the live run already did.

**Updated G3 status:**

| G3 condition | Status |
|---|---|
| No known leakage | **Pass.** Training and calibration use earlier races only (tested). |
| Post-quali beats the grid on walk-forward | **Pass** over all 91 races (CI excludes zero). Better on average in 2025–26 and 2026; at those sample sizes the CI touches zero. |
| Probabilities are calibrated | **Partial.** Post-quali winner: yes (within ±10 points in every window). Pre-quali favourites and points-finish probabilities: not yet (EXP-007 follow-up). |
| Live shadow record published | **Open** (coordinator) |

**Overall: G3 is close.** Open items: calibrate pre-quali favourites and points-finish probabilities; fix or retire the post-sprint stage; retrain the live models (the next `race_weekend.py before`); and the coordinator's live shadow record.

| Claim | OK now? |
|---|---|
| "Beats the starting grid after qualifying" | **Yes**, over 91 out-of-sample races (state the sample) |
| "Well-calibrated win chances after qualifying" | **Yes** (small buckets; keep checking live) |
| "Sharper than the odds" | **No** (and not a product claim anyway) |

## Workstream F: G3 model work (2026-11-14)

**Branch:** `claude/g3-model-work`. These are experiments only: the live recipe and `ml/models/` are unchanged during the 2026 shadow races, and anything adopted goes in after 6 Dec with owner approval. The walk-forward was regenerated on data through 2026 R20 (92 races): post-quali log-loss 1.149 vs grid 1.482, pre-quali 1.677 vs prior form 2.406. `pytest ml/tests`: 129 passed, 1 skipped.

| Open G3 item | Result | Evidence |
|---|---|---|
| Points-finish (top-10) calibration | **Fixed in the experiment; recommend.** The "ramp" sampler gives a softer midfield and leaves win chances unchanged. Post-quali points log-loss is 0.459 vs the grid-slot table 0.507 (2025–26: 0.480 vs 0.508; 2026: 0.461 vs 0.514), and every interval excludes zero. Buckets are within ~2 points. | [`EXP-008`](experiments/EXP-008-points-calibration.md) |
| Pre-quali favourite over-confidence | **Partly explained, small fix optional.** Caused by 2022–23 races in the temperature fit. Recency weighting (half-life 12) helps by 0.01–0.024, inside the noise. | [`EXP-009`](experiments/EXP-009-prequali-favourites.md) |
| Post-sprint stage regression | **Retire the stage.** The post-quali model is better on sprint weekends: 1.178 vs 1.524 (grid 1.266, 12 weekends). | [`EXP-010`](experiments/EXP-010-post-sprint-stage.md) |

### G3 status after workstream F

| G3 condition | Status |
|---|---|
| No known leakage | **Pass.** Every new calibrator is fitted on earlier races only (tested in `ml/tests/test_calibration.py`). |
| Post-quali beats the grid on walk-forward | **Pass** (1.149 vs 1.482 over 92 races). |
| Probabilities are calibrated | **Pass after qualifying once EXP-008 is wired in** (win: already; points: EXP-008). **Pre-quali: bands only** (win chances within ±10 points over all races, but recent favourites run a little hot). |
| Live shadow record published | **In progress.** Las Vegas 22 Nov, Qatar 29 Nov, Abu Dhabi 6 Dec. |

**To pass G3 by 2027-01-31:**
1. ~~Owner approves EXP-008 and EXP-010~~ Done 2026-11-14. EXP-009 not adopted.
2. After 6 Dec, wire them in through `train_stage_bundle`: store points outcomes in `reports/oos_calibration_history.parquet`, use `sequential_pl_ranks` in `ranking.simulate_race_draws`, stop selecting `post_sprint`.
3. Regenerate `EVAL.md` and check that EXP-008's numbers reproduce.
4. Publish the three-race shadow record.

### Claims after workstream F (once EXP-008 is wired in)

| Claim | OK? |
|---|---|
| "Calibrated chance of scoring points after qualifying" | **Yes** (state the sample: 92 races) |
| "Chance of scoring points before qualifying" | Bands only, no precise numbers |
| "Helps you pick a better Fantasy team" | **Still no.** Not re-tested here; EXP-007 stands. |
