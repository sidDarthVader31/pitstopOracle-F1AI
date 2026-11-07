# Pitstop Oracle — Phase 1 Forecast Integrity & Model Improvement Plan

> **Purpose:** Give Codex an implementation-ready plan for improving Pitstop Oracle's pre-qualifying and post-qualifying race forecasts without replacing the existing architecture prematurely.
>
> **Working principle:** Audit first, make small changes, test them, and measure every model change against a reproducible baseline. Do not add complexity simply because a feature or model is interesting.

## 1. Product objective

Pitstop Oracle should produce a predicted finishing-position distribution for every driver in a Grand Prix, at two distinct forecast times:

1. **Pre-qualifying:** after the final relevant practice session and before qualifying begins.
2. **Post-qualifying:** after qualifying is complete and the official starting grid, including penalties, is available.

For sprint weekends, session cutoffs must be explicit. A post-qualifying forecast must not silently include sprint results if it is intended to be generated before the sprint. Treat forecast timing as a data contract, not just a UI label.

The product should eventually show, for every driver:

- Probability of each finishing position (P1, P2, ... Pn).
- Expected finishing position.
- Win, podium and points probabilities.
- DNF probability as a separate outcome/risk indicator.

The existing Plackett–Luce simulation produces an ordering among drivers. It currently assigns every driver a position from P1 to Pn, so a true separate DNF/non-classified outcome is a future modeling decision. Do not claim the current distribution fully models retirement outcomes until that is addressed and evaluated.

## 2. Existing architecture — preserve before replacing

The repository currently includes:

- `ml/f1ml/ingest.py`: Jolpica historical data, Open-Meteo archived weather, plus canonical live-session/grid/weather sync.
- `ml/f1ml/features.py`: driver-race feature table, rolling statistics, practice pace, ratings and weather merge.
- `ml/f1ml/specs.py`: separate pre-quali and post-quali feature lists.
- `ml/f1ml/ratings.py`: shifted car/driver pace and DNF ratings, including a constructor reset at the 2026 regulation era.
- `ml/f1ml/ranking.py`: HistGradientBoosting rank regressor, DNF classifier, Plackett–Luce sampling and `WeekendRanker`.
- `ml/f1ml/train.py`: model training, calibration and walk-forward evaluation.
- `ml/f1ml/predict.py`: current-weekend feature assembly and model inference.
- `ml/f1ml/adapters/openf1.py`: practice/race pace and starting-grid ingestion.
- `ml/f1ml/adapters/openmeteo.py`: live weather forecast adapter.
- `ml/run_pipeline.py`: pipeline orchestration.

Do not replace the existing HistGradientBoosting + DNF + Plackett–Luce approach with a different model as the first step. First verify the correctness of its inputs and evaluation.

## 3. Non-negotiable forecast data contracts

Every feature used for a prediction must have been available at that prediction's cutoff.

### 3.1 Pre-qualifying snapshot

Allowed only when available before qualifying starts:

- Driver and constructor performance from earlier completed races.
- Championship standings entering the current event.
- Historical driver/circuit performance.
- Circuit attributes known before the event.
- Current-weekend practice information from sessions completed before the cutoff.
- Weather forecast information captured at or before the cutoff.
- Calendar-known context such as circuit, round, regulation era and sprint-weekend format.

Forbidden:

- Current-event qualifying results or qualifying-derived gaps.
- Official starting grid or penalties finalized after the cutoff.
- Current-event sprint results unless the chosen cutoff is after the sprint (which would be a different forecast definition).
- Current-event race pace, race laps, race result, pit stops or actual race-day weather.

### 3.2 Post-qualifying snapshot

Allowed only after qualifying and grid penalties are finalized:

- All valid pre-qualifying information.
- Current-event qualifying position and time gaps.
- Official starting grid and grid-vs-qualifying movement.

Sprint results are allowed only if the forecast is explicitly generated after the sprint. If the product wants a post-qualifying-before-sprint forecast, sprint-derived features remain forbidden.

Forbidden:

- Current-event race-session data, race result, actual race weather, race pit stops or any later session data.

### 3.3 Snapshot metadata

For each forecast, persist or make reproducible:

- `season`, `round`, `driver_id`, `forecast_stage`.
- `forecast_cutoff_at` in UTC.
- Session/cutoff definition (for example `after_fp3_before_quali`, `after_quali_before_sprint`, `after_quali_after_sprint`).
- Source and source timestamp for mutable inputs such as weather and starting grid.
- Model version and feature-spec version.
- The feature snapshot used for the prediction.

Never overwrite a previously issued forecast when later information arrives. A new forecast is a new snapshot/version.

## 4. Critical audit items — do these before model changes

Codex must inspect the current implementation and write findings before editing production logic.

### A. Audit every feature for temporal leakage

Create `docs/FEATURE_AUDIT.md` with one row per model feature and these columns:

- Feature name.
- Source table/API and source columns.
- Exact transformation.
- Earliest time the value is knowable.
- Whether it is allowed pre-quali.
- Whether it is allowed post-quali.
- Missing-data behavior.
- Leakage risk / notes.
- Test covering the cutoff.

**Known high-priority issues to investigate:**

1. `race_pace_delta` is built from OpenF1 rows where `session_type == "Race"`, merged onto the same event's driver row in `_merge_fp_pace()`, and currently appears in `FEATURE_PRE_QUALI_NUM`. This is a likely direct race-session leakage into both forecast stages. It must not be used for a pre-race forecast. Remove it from pre-quali features and ensure it is not used post-quali either. If retained as a historical feature, calculate a shifted prior-race summary and name it accordingly (for example `prior_race_pace_delta`).
2. Historical `precipitation_mm` / `is_wet` originate from archived observed weather, but live upcoming events use forecast weather. A historical pre-quali training row must not use actual race-day weather if the live model only has a forecast. Investigate archived forecast snapshots. Until they are available, remove actual event weather from pre-quali training or run a clear ablation showing its effect; do not quietly mix actual and forecast weather.
3. `fp1_best_lap_delta`, `fp2_best_lap_delta`, `fp3_best_lap_delta`, `fp_best_delta`, and `fp_longrun_delta` need session-aware eligibility. Confirm the session existed and was completed by the relevant forecast cutoff, particularly for sprint weekends and historical calendar changes.
4. `sprint_position` is in the post-quali feature list. It must only be used for forecasts whose cutoff is after the sprint. For a post-quali-before-sprint forecast, exclude it.
5. Current-event qualifying features such as `quali_gap_s`, `quali_vs_teammate`, and `teammate_quali_gap_s` are legitimate post-quali inputs but forbidden pre-quali. Verify no indirect copies enter shared features.
6. `pit_stops_avg` and `pit_time_avg_s` must be confirmed as strictly shifted historical summaries. Current-event pit stops are not available before the race.
7. `car_pace_ewma`, `driver_skill_ewma`, DNF hazard features, rolling driver/constructor statistics and circuit summaries must be checked for strict prior-event semantics, ordering, season boundaries, and reset behavior.
8. Verify the driver/circuit historical feature and constructor aggregates are shifted before the current event and that rows from the same race cannot leak into one another.

Do not merely remove every suspicious feature. Establish whether it is a valid historical feature, a current-session feature, or a future-session feature, then encode that distinction in the feature spec and tests.

### B. Audit the training population and DNF labels

Inspect `_load_training_frame()` and the DNF training path.

- It currently filters to rows with a valid numeric `position` before `train_dnf_model()` receives the frame.
- Determine whether non-classified/retired drivers are excluded by that filter.
- Print and document counts by `status`, `position` nullability and `dnf` label.
- Confirm the DNF classifier receives both positive and negative examples and that its labels match the intended target.
- Keep race classification position and DNF/non-classified outcome conceptually distinct.
- Do not silently train a DNF model on a population from which all or most DNF examples have been removed.

If a correction is required, separate the ranking-model training population from the DNF-model training population. Add tests proving positive and negative DNF examples reach the classifier.

### C. Audit walk-forward evaluation

- Confirm each test race is scored only using data and training labels from earlier races.
- Keep all drivers from one Grand Prix together in the same fold.
- Confirm calibration uses only a validation window strictly before the test race.
- Keep 2026 (new regulation era) performance reported separately.
- Label in-sample metrics as diagnostics only; do not present them as expected future performance.
- Review metric compatibility: the fast walk-forward path uses an analytic podium approximation, while full inference uses Monte Carlo Plackett–Luce sampling. Document this difference and ensure primary metrics are computed consistently.
- Check that the saved metrics JSON does not label training-set values as test metrics. In particular, inspect legacy/back-compat keys in `train.py` that currently populate `test_*` values from training metrics.
- Compare the model against simple, reproducible baselines on exactly the same out-of-sample races:
  - Pre-quali: a defined practice-pace / prior-form baseline.
  - Post-quali: starting-grid baseline and qualifying-gap baseline.
- Report the number of races and date/season range behind every metric.

## 5. Target dataset design

Do not immediately replace the existing parquet layout. First document how it maps to this logical dataset.

Logical grain:

`one row = one driver × one Grand Prix × one forecast stage`

Suggested columns:

| Group | Example fields |
|---|---|
| Identity | `season`, `round`, `driver_id`, `constructor_id`, `circuit_id` |
| Forecast definition | `forecast_stage`, `forecast_cutoff_at`, `session_cutoff` |
| Prior form | shifted driver/constructor rolling stats, prior pace ratings, prior DNF rates |
| Circuit | circuit characteristics and prior circuit-level history |
| Practice | only sessions available by cutoff, relative pace and long-run pace |
| Qualifying | post-quali only: position, gaps, official grid and penalty movement |
| Weather | forecast snapshot available at cutoff, with source timestamp |
| Targets | official classified finish position, winner/podium/points labels, separate DNF label |

The target columns must never be part of the feature matrix.

It is acceptable to keep shared historical tables and build separate pre/post feature views, as long as the cutoff is enforced and tested. Avoid duplicating the entire ingestion pipeline unnecessarily.

## 6. Circuit and car-performance features — later milestone

Only begin this after the leakage and data-quality gates pass.

### Circuit features

Prefer measurable numerical features over manually assigning tracks to labels:

- High-, medium- and low-speed corner proportions.
- Straight length / straight-line opportunity.
- Overtaking difficulty or a carefully defined historical proxy.
- Track width where a trustworthy source exists.
- Elevation / altitude.
- Pit-lane time loss.
- Surface and tyre-degradation proxies where reliable data exists.
- Era-aware historical Safety Car / VSC frequency.

Do not treat Monaco, Mexico and Spa as interchangeable examples of a simple circuit category. Their characteristics differ in important ways.

### Car / constructor traits

Separate direct measurements from inferred estimates. Candidate indicators:

- Straight-line speed / speed-trap performance.
- Relative sector pace by corner-speed profile.
- High-, medium- and low-speed corner performance.
- Braking performance proxies.
- Tyre degradation / long-run pace.
- Reliability / DNF rate.
- Aero efficiency proxies, only if the data supports them.

Use relative, within-event or within-season measures where possible so raw lap times are not confused by track length, weather, tyre or regulation differences.

Potential interactions to test later:

- `grid_position × overtaking_difficulty`
- `current_season_relative_pace × high_speed_corner_ratio`
- `straight_line_speed_index × straight_length_ratio`
- `low_speed_corner_index × low_speed_corner_ratio`

Do not create one independent model per circuit. Start with circuit attributes as features and test whether they improve walk-forward performance.

## 7. 2026 regulation change and concept drift

Treat 2026 as a new regulation era, not just another season.

- Keep the existing era indicator/reset logic under test.
- Circuit characteristics transfer across eras better than constructor/car pace.
- Driver skill may transfer partially; car-performance history should be treated more cautiously after a regulation reset.
- Do not hard-code arbitrary percentage weights for old versus new seasons.
- Use walk-forward evaluation and ablation tests to determine how much prior-era data helps.
- Report a 2026-only transfer diagnostic: train on pre-2026 data, test on 2026. Also report the live-like expanding-window evaluation that incorporates only earlier 2026 races.
- Represent uncertainty honestly when little current-era data exists.

## 8. Model strategy

Preserve the current baseline architecture initially:

1. Pace/ranking model estimates relative driver performance.
2. DNF model estimates retirement risk.
3. Race-order simulation turns relative scores into a finishing-position distribution.

Do not introduce neural networks or deep learning in Phase 1.

Run controlled ablations, changing one thing at a time:

- Current model with all currently approved features.
- Without weather.
- Without current-weekend practice pace.
- Without DNF adjustment.
- With corrected DNF training population.
- With/without prior-regulation constructor history.
- With circuit features once available.

The fixed DNF score penalty (currently a constant multiplier) is a hypothesis, not a fact. Compare it with no DNF adjustment and alternative mappings using out-of-sample data. Do not tune it on the final test races.

The current Plackett–Luce sampler forces all drivers into a complete order. Design and evaluate an explicit DNF outcome treatment separately rather than assuming a score penalty fully models non-classification.

## 9. Evaluation and acceptance criteria

For both forecast stages, report:

- Mean absolute error (MAE) of expected finishing position.
- Rank correlation.
- Winner hit rate.
- Top-three coverage / hit rate.
- Win-probability log loss and Brier score.
- Calibration plots / bins.
- Position-distribution quality (for example, per-position probability calibration or a suitable ranked-probability score once implemented).
- DNF classification metrics and calibration, evaluated on a valid population.

Evaluation rules:

- Primary evaluation is chronological walk-forward.
- Group by Grand Prix, never split drivers from one race across train and test.
- All feature values must respect the forecast cutoff.
- Keep 2026 results separate.
- Compare against baselines on the exact same races.
- Record model version, feature list, training window, evaluation window and random seed.
- Do not promote a candidate model just because it improves in-sample performance or one race.

A candidate feature/model change is accepted only if:
1. Its data availability is valid for the forecast cutoff.
2. Tests pass.
3. It improves a pre-declared out-of-sample metric or provides a justified product benefit without materially damaging other important metrics.
4. The comparison is documented in an evaluation report.

## 10. Retraining and race-weekend operating procedure

**Do not retrain the model before every race by default.** Training and prediction are separate operations.

Recommended lifecycle:

### Before / during a race weekend

1. Ingest new practice, weather forecast and (when published) starting-grid data.
2. Build the current event's feature snapshot for the relevant forecast cutoff.
3. Load the currently approved model artifact.
4. Generate pre-quali or post-quali predictions.
5. Save the forecast snapshot and model version.

This is inference, not training. It should not require retraining.

### After a race

1. Ingest official race results and other completed-session data.
2. Rebuild historical features so the completed race becomes available to future events.
3. Run data-quality and evaluation checks.
4. Optionally train a **candidate** model using the newly completed race.
5. Compare candidate versus the currently approved model using walk-forward validation and agreed guardrails.
6. Promote the candidate only when the evidence supports it. Keep the previous model available for rollback.

### Initial cadence

For Phase 1, use this conservative policy:

- Refresh weekend inputs whenever new eligible sessions/data arrive.
- Do not train between practice, qualifying and race sessions just to make a prediction.
- After each completed Grand Prix, refresh the dataset and run evaluation.
- Candidate retraining can be run after each Grand Prix initially, but automatic promotion is disabled.
- Promote manually after reviewing accumulated evidence; if the dataset is small or results are noisy, wait for several races before promotion.
- Later, automate candidate training on a schedule (for example after every 2–3 completed races) if it proves useful.

A new race adds training data, but one race does not necessarily provide enough evidence that a model has improved. Model versioning and controlled promotion matter more than retraining frequency.

## 11. Implementation milestones for Codex

### Milestone 0 — Repository and data reconnaissance

- Read the listed source files and relevant tests.
- Identify actual data paths and dependency configuration.
- Inspect the current parquet schemas and counts if local data is available.
- Do not assume README metric values are current or correct.
- Produce a short findings summary before code edits.

**Deliverables:** repository map, dataset schema summary, initial risk list.

### Milestone 1 — Feature/cutoff audit

- Create `docs/FEATURE_AUDIT.md`.
- Trace every feature in `FEATURE_PRE_QUALI_NUM` and `FEATURE_POST_QUALI_EXTRA`.
- Classify each as prior-history, current practice, qualifying, sprint, grid, weather or race-result-derived.
- Flag direct and indirect leakage.
- Propose the minimum safe feature-spec changes.

**Deliverables:** completed audit and explicit feature allowlists per forecast cutoff.

### Milestone 2 — Data and leakage tests

Add automated tests for:

- A current race's race-session pace cannot enter pre-quali or post-quali features.
- Current-event actual weather cannot enter pre-race predictions.
- Qualifying-derived features are absent from pre-quali feature matrices.
- Sprint results are absent before the sprint cutoff.
- Grid data is absent before the official grid is available.
- Rolling driver/constructor/circuit features use prior events only.
- Pit-stop aggregates use prior races only.
- Driver rows from a single race stay in the same validation fold.
- DNF positive and negative labels are present in the DNF training population.
- Missing practice/session values are handled explicitly, not silently interpreted as pace.

Use small synthetic DataFrames for deterministic unit tests. Do not require network calls in unit tests.

### Milestone 3 — Correct the training/evaluation data paths

- Separate ranking and DNF training populations if necessary.
- Remove or shift any invalid future-session feature.
- Ensure historical training data mimics the data actually available at each live cutoff.
- Fix misleading metric keys that currently map training metrics into legacy `test_*` fields.
- Make walk-forward scoring and final inference use compatible calculations for primary metrics.
- Document any temporary data limitations.

### Milestone 4 — Baseline report

- Re-run walk-forward evaluation after corrections.
- Report pre- and post-quali separately.
- Compare with the same-race baselines.
- Include 2026 separately.
- Include the number of races and date range.
- Save the results in `ml/reports/EVAL.md` and `ml/reports/metrics.json`.

Do not add circuit features until this corrected baseline is established.

### Milestone 5 — Circuit/car feature research and incremental additions

- Identify reliable public data sources and availability for circuit geometry and performance indicators.
- Build a source/coverage inventory before implementation.
- Add a small set of high-confidence features first.
- Run feature ablations and keep only features that improve out-of-sample performance or uncertainty quality.

### Milestone 6 — Forecast snapshots and model lifecycle

- Save immutable prediction snapshots with cutoff, model and feature versions.
- Separate data refresh, feature generation, inference, candidate training and model promotion commands.
- Keep champion model artifacts stable until a candidate passes evaluation.
- Document how to roll back.

## 12. Codex execution instructions

Work incrementally and do not implement all milestones in one large change.

For each milestone:

1. Inspect the relevant files and tests before editing.
2. Explain the proposed change and the reason in plain language.
3. Implement the smallest coherent change.
4. Add or update tests.
5. Run the relevant test suite and report exact commands/results.
6. Summarize changed files, assumptions, remaining risks and next step.
7. Do not claim tests passed unless they were actually run.
8. Do not fetch external APIs in tests.
9. Do not delete existing model artifacts or data.
10. Do not commit directly to the default branch unless explicitly asked. Work on a feature branch and provide a reviewable diff.

**First task for Codex:** complete Milestone 0 and Milestone 1 only. Do not change model architecture or add circuit/car features yet. After presenting the feature audit and the exact proposed code changes, stop for review before implementing Milestone 2.

## 13. Definition of Phase 1 complete

Phase 1 is complete when:

- Pre- and post-quali forecasts have explicit, tested cutoffs.
- No future-session or actual-weather leakage remains in historical training features.
- DNF training includes a valid and documented target population.
- Walk-forward evaluation is reproducible and compared to relevant baselines.
- Both forecast stages produce a finishing-position distribution for every eligible driver.
- Forecasts are reproducible from saved feature snapshots and model versions.
- Model retraining and promotion are separate, documented operations.
- Performance claims are based on out-of-sample evidence, not in-sample metrics.

---

## Beginner-friendly note

This project uses machine learning, but the immediate challenge is not choosing a more sophisticated algorithm. It is ensuring that the model learns from the same kind of information it would actually have before a race. A simple model trained on correctly timed data is more useful than a sophisticated model trained with information from the future.
