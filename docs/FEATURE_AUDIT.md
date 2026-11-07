# Feature / cutoff audit (Phase 1, Milestones 0–1)

This document is the Milestone 0 reconnaissance plus the Milestone 1 feature/cutoff audit required by [`PHASE_1_FORECAST_PLAN.md`](PHASE_1_FORECAST_PLAN.md). **No production model code was changed.** Proposed edits for Milestones 2–3 are listed at the end for review.

Counts below were taken from local parquet on `codex/phase1-forecast-plan` (processed table dated with 2022–2026 through Mexico City 2026 R19). Do not treat README hit rates as current.

---

## Milestone 0 — repository map

| Path | Role |
|---|---|
| [`ml/run_pipeline.py`](../ml/run_pipeline.py) | Default path: ingest → `build_driver_race()` → **`train_and_eval()`**. Weekend refresh currently retrains. |
| [`ml/f1ml/ingest.py`](../ml/f1ml/ingest.py) | Jolpica raw history; Open-Meteo **archive** weather; OpenF1 / forecast canonical sync. |
| [`ml/f1ml/adapters/openf1.py`](../ml/f1ml/adapters/openf1.py) | FP1–FP3 + **Race** pace into `fp_pace.parquet`; starting grid. |
| [`ml/f1ml/adapters/openmeteo.py`](../ml/f1ml/adapters/openmeteo.py) | Live **forecast** for future race dates only. |
| [`ml/f1ml/features.py`](../ml/f1ml/features.py) | One row per driver-race; rolling form; FP/race pace merge; pit shift; observed weather + forecast overlay for unfinished races. |
| [`ml/f1ml/specs.py`](../ml/f1ml/specs.py) | `FEATURE_PRE_QUALI_NUM` + `FEATURE_POST_QUALI_EXTRA`. Only two modes: `pre_quali`, `post_quali`. |
| [`ml/f1ml/ratings.py`](../ml/f1ml/ratings.py) | Shifted car/driver EWMA + DNF hazard. Constructor reset at 2026. Current-row ratings do **not** include the current race update. |
| [`ml/f1ml/ranking.py`](../ml/f1ml/ranking.py) | HGB rank regressor, HGB DNF classifier, Plackett–Luce MC, DNF score penalty `* 3.0`. |
| [`ml/f1ml/train.py`](../ml/f1ml/train.py) | Shared training frame; walk-forward; in-sample diagnostics; writes `metrics.json` / `EVAL.md`. |
| [`ml/f1ml/predict.py`](../ml/f1ml/predict.py) | Upcoming-race assembly + weekend card (MC inference). |
| [`ml/tests/`](../ml/tests/) | Ranking, weekend overlay, pit merge, Jolpica cache. **No cutoff/leakage tests** for race pace, weather, sprint, DNF labels, or folds. |

Data layout matches [`DATASET.md`](DATASET.md): `data/raw/*` (Jolpica + archive weather), `data/canonical/*` (OpenF1 / forecast), `data/processed/driver_race.parquet`.

---

## Milestone 0 — dataset snapshot

| Table | Rows | Races / notes |
|---|---|---|
| `raw/races.parquet` | 115 | 2022–2026 calendars (2026 has 23 rounds) |
| `raw/results.parquet` | 2256 | 111 completed GPs (2026 R01–R19) |
| `raw/qualifying.parquet` | 2251 | 111 races |
| `raw/sprints.parquet` | 552 | 27 sprint weekends (2026: rounds 2, 4, 5, 9, 12, 17) |
| `raw/weather.parquet` | 115 | Archive **observed** precip for every calendar round |
| `raw/pitstops.parquet` | 3053 | 111 races |
| `canonical/fp_pace.parquet` | 10260 | 2023–2026 only; includes `Race` and `*_LONGRUN` |
| `canonical/weather_forecast.parquet` | 1 | 2026 R20 Brazil only |
| `canonical/starting_grid.parquet` | 1 | Stale 2026 R13 Antonelli override |
| `processed/driver_race.parquet` | 2256 / 111 races / 78 cols | Every completed race has a winner and **non-null `position`** |

Upcoming on calendar, not in the feature table: 2026 R20 Brazil (2026-11-08), R21 Las Vegas, R22 Qatar, R23 Abu Dhabi.

OpenF1 session mix: FP1 1481, FP2 1113, FP3 1077, Race 1459 (77 races). 2022 has **no** FP/race pace.

---

## Milestone 0 — initial risk list

Severity is for forecast integrity, not product polish.

1. **Critical — current-event race pace in both model specs.** `race_pace_delta` is merged from OpenF1 `session_type == "Race"` in `_merge_fp_pace()` and listed in `FEATURE_PRE_QUALI_NUM` (hence also post-quali). Coverage 64.5% (1456/2256). Correlation with finish position ≈ 0.30. Live upcoming assembly does **not** overlay Race pace, but it copies the previous race row and never clears `race_pace_delta`, so inference can carry the **previous** event’s race pace under a current-event name.
2. **Critical — DNF label is not “retirement.”** `_is_classified()` only accepts `Finished` and `+N Lap`. Jolpica `Lapped` (414 rows) is labeled `dnf=1`. Training DNF rate is **33.4%** (754/2256). After treating `Lapped` as classified, positives fall to **340**. The plan’s worry that `position.notna()` drops DNFs is **false on this extract** (0 null positions). The real bug is over-labeling classified lapped cars as DNF. Ranker and DNF still share `_load_training_frame()`; that split remains useful once labels are fixed.
3. **High — train/serve weather mismatch.** Historical rows always use archive **race-day observed** precip (`is_wet` if precip ≥ 1 mm → 43/111 races wet, 38.7%). Live upcoming uses forecast overlay. There is **no** historical forecast archive. Training the wet flag on actuals then inferring from forecasts is leakage relative to the live cutoff.
4. **High — sprint is a silent post-quali feature.** `sprint_position` is in `FEATURE_POST_QUALI_EXTRA` with no cutoff. `has_sprint` is `sprint_position.notna()` (result-derived, not calendar). Live overlay sets sprint only when sprint parquet exists; historical post-quali training always sees sprint results on sprint weekends.
5. **High — eval protocol ≠ inference.** Walk-forward uses `_score_bundle_fast` (softmax win probs + podium heuristic `win_prob * 2.8 + (1 - dnf) * 0.08`). `_eval_mode` / live `weekend_card` use Monte Carlo Plackett–Luce. Shipping models are trained with `val=None` so **temperature stays 1.0**.
6. **High — walk-forward is not every race except 2026.** `_walk_forward_folds` scores season-opener only for 2023–2025, then every 2026 race. `EVAL.md` “full walk-forward” looks race-by-race but pre-2026 is one fold per season.
7. **Medium — `test_*` keys half-fixed.** `test_pre_quali` / `test_post_quali` come from 2026 OOS. `test_pole_hit`, `test_champ_hit`, `test_rf_top3` are still copied from **in-sample** `train_post_quali_*`. README pole **69.4%** is not the current 2026 walk-forward number (post-quali hit **36.8%**, in-sample pole **59.5%**).
8. **Medium — practice missingness.** FP3 missing 52%, FP2 51%, FP1 35%. Sprint weekends often have no FP3. `SimpleImputer(median)` treats missing as “typical pace.” Upcoming path sets `fp_longrun_delta = fp_best_delta` (not true long-run). `_merge_fp_pace` rebuilds `fp_longrun_delta` per session and **keeps the last session in the loop** (FP3), then fills from `fp_best_delta`.
9. **Medium — `circuit_grid_to_finish_delta` fillna(0.0)** encodes “no overtaking” when history is thin. Upcoming `circuit_avg_finish_prior` averages **all** historical visits on that circuit (OK for a future GP; not the same transform as the shifted expanding mean used in training).
10. **Low / confirm-safe — shifted history.** Rolling driver/constructor stats, pit aggregates (`shift(1).rolling(5)`), `teammate_quali_h2h_rate`, `circuit_avg_finish_prior` (training path), and ratings assignment-before-update look prior-event. Round-1 2023 pit stats can be non-null because the driver window crosses seasons — that is allowed history, not current-race pit stops.
11. **Process — default pipeline retrains.** `python run_pipeline.py` always calls `train_and_eval()`. That contradicts the Phase 1 operating rule (ingest + infer on weekends; promote models separately).

---

## Cutoff definitions used in this audit

| `forecast_stage` | Intended cutoff | Allowed current-weekend inputs |
|---|---|---|
| `pre_quali` | After last relevant practice, before quali (sprint: after FP1, before sprint quali / shootout) | Prior races; standings entering the event; completed FP sessions for that format; weather **forecast** as-of cutoff |
| `post_quali_before_sprint` | After qualifying + published grid/penalties, **before** sprint | Pre-quali set + quali + grid |
| `post_quali_after_sprint` | After sprint race | Previous + `sprint_position` |
| `post_quali` (current code) | Ambiguous mix of the two post-quali rows | Quali + grid + sprint whenever present |

The code today only has `pre_quali` / `post_quali`.

---

## Feature table

Columns: **Feature** · **Source** · **Transform** · **Earliest knowable** · **Pre-quali** · **Post-quali** · **Missing** · **Leakage / notes** · **Test**

Legend: **OK** = allowed at that cutoff if built as specified. **NO** = forbidden. **COND** = allowed only for a narrower cutoff.

### A. `FEATURE_PRE_QUALI_NUM`

| Feature | Source | Transform | Earliest knowable | Pre | Post | Missing | Leakage / notes | Test |
|---|---|---|---|---|---|---|---|---|
| `car_pace_ewma` | ratings from prior `quali_gap_s`, `fp_longrun_delta`, `race_pace_delta` | Shifted EWMA; constructor reset 2026; **assigned then updated** | After previous race | OK | OK | New constructor → `0.0` | Current-event race/quali enter the **next** race’s rating, not this row. Do not confuse with raw `race_pace_delta`. Upcoming uses `latest_ratings_before_round`. | `test_ratings_no_leakage_from_current_race` (weak: checks change, not exact prior-only) |
| `driver_skill_ewma` | prior `teammate_quali_gap_s`, `teammate_race_gap` | Shifted EWMA; carries across 2026 | After previous race | OK | OK | New driver → `0.0` | `teammate_race_gap` uses current **result** only to update future skill. | same |
| `car_driver_combo` | sum of the two EWMAs | Additive | After previous race | OK | OK | 0 | Same as parents. | none |
| `champ_position_before` | `standings.parquet` `after_round = round-1` | Merge; R1 → null position, 0 points/wins | After previous GP | OK | OK | R1 null | Correct prior standings. | none |
| `champ_points_before` | same | same | After previous GP | OK | OK | R1 = 0 | OK | none |
| `champ_wins_before` | same | same | After previous GP | OK | OK | R1 = 0 | OK | none |
| `driver_id_points_last_3` | `results.points` | `shift(1).rolling(3).sum()` per driver | After previous GP | OK | OK | Early-career null | Prior-only in training. Upcoming uses `prior.tail(3)` in-season only (weaker than full cross-season roll). | none |
| `driver_id_points_last_5` | same | window 5 | After previous GP | OK | OK | null | same | none |
| `driver_id_avg_finish_last_5` | `position` | `shift(1).rolling(5).mean()` | After previous GP | OK | OK | null | Uses classified position including DNFs that still have a rank. | none |
| `driver_id_dnf_rate_last_5` | `dnf` | `shift(1).rolling(5).mean()` | After previous GP | OK | OK | null | **Inherits Lapped-as-DNF bug.** | none |
| `constructor_id_points_last_3` | summed constructor points per race | shift+roll on constructor | After previous GP | OK | OK | null | Prior-only. | none |
| `constructor_id_points_last_5` | same | window 5 | After previous GP | OK | OK | null | OK | none |
| `constructor_id_avg_finish_last_5` | constructor best finish | shift+roll | After previous GP | OK | OK | null | OK | none |
| `constructor_id_dnf_rate_last_5` | constructor mean `dnf` | shift+roll | After previous GP | OK | OK | null | Inherits DNF label bug. | none |
| `dnf_hazard_car` | shifted EWMA of constructor `dnf` | assigned then updated; clip | After previous GP | OK | OK | prior 0.15 | Label bug. Era reset with car pace. | `test_era_reset_constructor_prior` |
| `dnf_hazard_driver` | shifted EWMA of driver `dnf` | same | After previous GP | OK | OK | prior 0.10 | Label bug. | none |
| `dnf_hazard_combined` | `1-(1-car)*(1-driver)` | clip 0.02–0.90 | After previous GP | OK | OK | ~0.24 | Label bug. | none |
| `circuit_avg_finish_prior` | `position` by `(driver_id, circuit_id)` | `shift(1).expanding().mean()` in training | After driver’s previous visit | OK | OK | 34.7% null (first visit) | Training path is shifted. Upcoming uses **unshifted** mean of all history for that circuit (includes only completed GPs, so OK for a future event). | none |
| `circuit_grid_to_finish_delta` | `grid - position` | `shift(1).expanding(min_periods=3).mean()` by `circuit_id`; **fillna(0)** | After 3 prior races at circuit | OK* | OK* | filled 0 | 0 means “unknown overtaking = none.” Prefer null + impute or a dedicated missing flag. Upcoming uses full-history mean or 0. | none |
| `teammate_quali_h2h_rate` | prior `beat_teammate_quali` | `shift(1).rolling(5).mean()` | After previous quali | OK | OK | 1.5% null | Uses **prior** quali H2H only — allowed pre-quali. | none |
| `is_wet` | archive daily precip ≥ 1 mm; forecast overlay if race unfinished | int | Forecast: before event. Actual: race day | **NO** (actual) / COND (forecast) | same | filled False | Training = observed race-day weather. Live upcoming = forecast. 38.7% of races flagged wet (threshold is aggressive vs “wet race”). | none |
| `has_sprint` | `sprint_position.notna()` before fill | 0/1 | Calendar known before weekend | COND | COND | 0 | Value is OK as calendar context; **derivation is result-based**. Should come from calendar / known sprint rounds. | none |
| `round` | calendar | int | Before weekend | OK | OK | never | OK | none |
| `precipitation_mm` | `weather.parquet` archive; forecast overlay for unfinished | mm | Forecast vs race day | **NO** (actual) / COND (forecast) | same | 0 (always filled historically) | Same train/serve mismatch as `is_wet`. | none |
| `fp1_best_lap_delta` | OpenF1 FP1 `best_lap_seconds` − session median | current event | After FP1 | COND | COND | 35% null | Allowed only if FP1 completed by cutoff. Median imputer fills missing as typical. No 2022 coverage. | `test_merge_fp_pace_fp1_fallback` (merge only) |
| `fp2_best_lap_delta` | OpenF1 FP2 | After FP2 | COND | COND | 51% null | Forbidden on sprint weekends that have no FP2, or if cutoff is after FP1 only. | none |
| `fp3_best_lap_delta` | OpenF1 FP3 | After FP3 | COND | COND | 52% null | Sprint weekends typically have no FP3 — must be null, not imputed as pace. | none |
| `fp_best_delta` | FP3 else FP2 else FP1 | After last completed FP | COND | COND | 28% null | Session-aware fallback is OK if each source honors cutoff. | same FP1 test |
| `fp_longrun_delta` | last `*_LONGRUN` merge (effectively FP3) then fill `fp_best_delta` | After that session | COND | COND | same as `fp_best_delta` | Training ≠ upcoming (`fp_longrun_delta = fp_best_delta` live). Race_LONGRUN unused. | none |
| `race_pace_delta` | OpenF1 **Race** best lap − race median | After the race | **NO** | **NO** | 35% null (no 2022; 77 races) | **Direct future-session leak into both specs.** Corr vs finish ≈ 0.30. Drop from specs. Optional later: shifted `prior_race_pace_delta`. | **none** |
| `pit_stops_avg` | pitstops count per driver-race | `shift(1).rolling(5).mean()` | After previous GP | OK | OK | 7.8% null | Confirmed shifted. R1 of later seasons may use prior-season pits. | `test_merge_pitstops_populates_without_duplicate_columns` (does not assert no current-race leak) |
| `pit_time_avg_s` | mean pit duration, same shift | After previous GP | OK | OK | same | OK | same |
| `regs_era` | `season >= 2026` | 0/1 | Before weekend | OK | OK | never | Calendar/regulation flag. | `test_era_reset_constructor_prior` |

### B. `FEATURE_POST_QUALI_EXTRA`

| Feature | Source | Transform | Earliest knowable | Pre | Post | Missing | Leakage / notes | Test |
|---|---|---|---|---|---|---|---|---|
| `quali_position` | Jolpica qualifying | current event | After quali | **NO** | OK | 0.3% | Legitimate post-quali. Verify pre-quali matrix never includes it (today `feature_columns("pre_quali")` does not). | none for cutoff |
| `quali_gap_s` | best quali − pole | current event | After quali | **NO** | OK | with quali | OK post-quali. Also feeds **future** `car_pace_ewma` only. | `test_quali_gap_baseline_not_pole_only` |
| `quali_gap_to_median_s` | vs field median | After quali | **NO** | OK | with quali | OK post-quali | none |
| `grid` | results.grid historically; OpenF1/manual starting_grid live | current event; 0 → null | After official grid | **NO** | OK | 0.7% | Canonical grid file is nearly empty (1 row). Historical grid is results grid (post-penalty). | weekend overlay test |
| `grid_vs_quali` | `grid - quali_position` | After grid | **NO** | OK | with both | OK | none |
| `best_quali_seconds` | quali | After quali | **NO** | OK | with quali | Absolute time is circuit-scale; gap features are safer. Keep for now (post only). | none |
| `quali_vs_teammate` | quali_position − teammate min | After quali | **NO** | OK | with quali | Current-event; correctly **not** in pre list. | none |
| `teammate_quali_gap_s` | quali seconds vs teammate | After quali | **NO** | OK | with quali | Current-event seconds H2H. Distinct from pre `teammate_quali_h2h_rate`. | none |
| `sprint_position` | `sprints.parquet`; fillna **−1** | current sprint | After sprint | **NO** | **COND** (`after_sprint` only) | −1 means no sprint / missing | −1 is a real numeric value the HGB can treat as “grid-like.” Must be absent before sprint. | none |

### C. Derived / ratings inputs (not all are model columns)

| Feature | Notes | Test |
|---|---|---|
| `dnf` | Target. `Lapped` → 1 is wrong. DNS/DSQ mixed with on-track retirements. | none (need population test) |
| `position` | Always present on this extract, including retirements (classified order). Ranker target `-position`. | none |
| `teammate_race_gap` | Current-race result; used only to update **next** driver skill. Must not be added to specs. | none |
| `grid_finish_delta` | Intermediate for circuit overtaking index. | none |
| `beat_teammate_quali` | Intermediate for H2H rate. | none |

---

## Allowlists (minimum safe)

### Pre-qualifying (`after last eligible practice, before quali`)

Keep:

- Ratings: `car_pace_ewma`, `driver_skill_ewma`, `car_driver_combo`, `dnf_hazard_*` (after DNF label fix)
- Form: champ_*_before, driver/constructor last-3/5 rolls, `teammate_quali_h2h_rate`
- Circuit history: `circuit_avg_finish_prior`, `circuit_grid_to_finish_delta` (prefer null vs 0)
- Context: `round`, `regs_era`, calendar `has_sprint`
- Pits: `pit_stops_avg`, `pit_time_avg_s`
- Practice: `fp1_*` always if session exists; `fp2_*` / `fp3_*` / `fp_best_delta` / `fp_longrun_delta` only if that session is completed **and** exists for the weekend format

Remove from the live/train pre-quali matrix:

- `race_pace_delta`
- event-day `is_wet` / `precipitation_mm` until a forecast-at-cutoff series exists (optional dated ablation vs drop)

### Post-qualifying before sprint

Pre-quali allowlist **plus** `quali_position`, `quali_gap_s`, `quali_gap_to_median_s`, `grid`, `grid_vs_quali`, `best_quali_seconds`, `quali_vs_teammate`, `teammate_quali_gap_s`.

Still forbidden: `sprint_position`, `race_pace_delta`, race-day actual weather.

### Post-qualifying after sprint

Previous plus `sprint_position` (null if the driver did not take part; do not use −1 as a sentinel if the imputer will invent pace-like values — prefer NaN).

---

## Proposed Milestone 2 tests (do not implement yet)

Add synthetic tests in `ml/tests/` (no network):

1. `test_race_pace_not_in_pre_or_post_feature_lists` — after spec change, `race_pace_delta` not in `feature_columns`.
2. `test_merge_fp_pace_does_not_attach_current_race_as_model_feature` — Race session may exist on disk; ranker matrix must not contain current-event race pace.
3. `test_historical_weather_not_used_as_forecast_proxy` — document chosen policy (drop vs ablation flag).
4. `test_pre_quali_matrix_excludes_quali_and_grid_and_sprint`.
5. `test_sprint_position_absent_before_sprint_cutoff`.
6. `test_grid_absent_before_official_grid`.
7. `test_rolling_form_and_pits_exclude_current_race` (strengthen existing pit test).
8. `test_walk_forward_keeps_race_drivers_together`.
9. `test_dnf_training_population_has_positives_and_negatives` **and** `test_lapped_status_is_classified`.
10. `test_missing_fp_stays_nan_not_zero` / not silently equal to field median inside the feature table (imputer may still fill at fit time).

---

## Proposed Milestone 3 code changes (do not implement yet)

Smallest coherent diffs after this review:

1. **[`ml/f1ml/specs.py`](../ml/f1ml/specs.py)**  
   - Remove `race_pace_delta` from `FEATURE_PRE_QUALI_NUM`.  
   - Split extras: `FEATURE_POST_QUALI_EXTRA` (no sprint) and `FEATURE_POST_SPRINT_EXTRA = ["sprint_position"]`.  
   - Optional: `WeekendMode` grows `"post_sprint"` **or** a cutoff flag on `feature_columns`.  
   - Weather: remove `is_wet` / `precipitation_mm` from the default pre-quali list (keep a commented ablation list).

2. **[`ml/f1ml/features.py`](../ml/f1ml/features.py)**  
   - Stop merging OpenF1 `Race` into the model feature table (keep a shifted `prior_race_pace_delta` later if an ablation wants it).  
   - Set `has_sprint` from calendar/sprint-round set, not `sprint_position.notna()`.  
   - `sprint_position`: NaN when no sprint, not −1.  
   - Fix `_is_classified`: treat `Lapped` (and other classified-but-lapped Jolpica strings) as classified.  
   - FP long-run: fallback FP3→FP2→FP1 longrun without dropping earlier columns.  
   - `circuit_grid_to_finish_delta`: stop `fillna(0.0)`.

3. **[`ml/f1ml/predict.py`](../ml/f1ml/predict.py)**  
   - When cloning the previous race for an upcoming GP, drop `race_pace_delta` and current-event quali/sprint/grid/weather before overlay.  
   - Do not copy previous `fp_*` into the new round.  
   - Overlay forecast weather only; never leave previous race wet flags.

4. **[`ml/f1ml/train.py`](../ml/f1ml/train.py)**  
   - Ranker population: classified finishers with numeric position (policy to decide: include lapped).  
   - DNF population: full completed-race rows with corrected labels (DNS/DSQ policy explicit).  
   - Walk-forward primary metrics via the same scorer as `WeekendRanker.predict_frame` (or document softmax as a diagnostic only).  
   - `test_pole_hit` / `test_champ_hit` / `test_rf_top3` from OOS, not train.  
   - Expand folds to every GP or clearly label season-opener diagnostics.

5. **[`ml/run_pipeline.py`](../ml/run_pipeline.py)** — later (M6): default weekend command = ingest + features, not train. Out of Milestone 3 unless it is a one-flag `--skip-train` (already have `--skip-ingest`; add `--skip-train` as a tiny safety).

Do **not** add circuit geometry, neural nets, or forecast snapshot stores in Milestone 3.

---

## What this audit is not claiming

- Ratings EWMAs are leak-free for the **current** row; they still **use** race pace to update the next row, which is intended.
- Pit and rolling form look correctly shifted; they still need tests.
- DNF examples **do** reach `train_dnf_model` today (754 positives) — but many are lapped cars, so the classifier is not learning retirement.
- Archived Open-Meteo **forecasts** for past races are not in-repo; collecting them is extra scope.

---

## Stop for review

Milestones 0 and 1 are complete. Next, after approval: Milestone 2 tests, then the spec/feature/train edits above, then a corrected walk-forward report (Milestone 4).
