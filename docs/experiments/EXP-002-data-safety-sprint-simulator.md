# EXP-002 — Data foundation, event risk, sprint cutoffs, and outcome simulation

**Status:** implemented where supported by local data; no candidate promoted
**Scope:** Roadmap Stages 1, 3, 4, 5, and 6 implementation follow-through
**Evaluation:** chronological event-risk diagnostic; feature and lifecycle behavior covered by unit tests.

## Changes

- Added a reproducible data inventory at `ml/audit_data.py` and `ml/reports/DATA_AUDIT.md`, documenting row grain, missingness, source timing, label counts, and unsupported circuit/car/weather fields.
- Added OpenF1 historical meeting, race-session, and race-control ingestion for 2023 onward. Derived race-level deployed SC/VSC labels while retaining source messages and timestamps.
- Added sourced circuit-type and shifted circuit-history features. Traits without measured evidence (corner-speed mix, sector/speed-trap car traits, elevation, pit-lane loss) remain missing/deferred.
- Made retirement DNF labels exclude DNS, DSQ, and withdrawals while treating lapped classifications as classified. Kept DNF labels distinct from numeric finishing-position training rows.
- Added `post_sprint` as a separate feature stage; sprint result enters only after the sprint. Pre-quali and post-quali-before-sprint remain separate.
- Updated race outcome Monte Carlo to sample DNF risk separately, rank classified drivers, and report conditional finishing-position probabilities/expected finish.
- Added versioned candidate-only training and immutable stage-specific forecast snapshots with model/feature metadata and source timestamps. Normal evaluation does not promote model artifacts.

## Safety-car/VSC diagnostic

OpenF1 race-control covered 89 race sessions from 2023–2026, with 40 SC-positive and 31 VSC-positive labels. An expanding logistic model using only prior circuit event rates, sourced circuit type, round, and regulation era was evaluated on 59 chronological races after a 30-race initial history.

| Event | Model log loss | Expanding-prevalence log loss | Model Brier | Prevalence Brier | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Safety Car | 0.861 | 0.699 | 0.308 | 0.253 | 0.349 | 0.397 |
| Virtual Safety Car | 0.736 | 0.642 | 0.262 | 0.224 | 0.340 | 0.509 |

Both candidates lost to the simple expanding-prevalence baseline on log loss and Brier. Their probabilities are not fed into production forecasts. No artifact was saved.

## Pace baseline after the full chronological refresh

The report now covers 91 eligible held-out races from 2022–2026, with an expanding chronological fit/calibration process. Across all eras, post-quali win log loss was 1.729 versus 1.466 for the official-grid baseline and 1.954 for the qualifying-gap baseline. In 2026 alone, the post-quali model scored 1.548 versus 1.609 for grid and 1.559 for qualifying gap. That 0.011 difference against qualifying gap on 19 races is too small a sample to justify promotion. The pre-quali model scored 2.743 overall versus 3.152 for prior form; in 2026, it scored 3.699 versus 3.125 for prior form. Retain the model as a diagnostic candidate and continue to report these same-cutoff comparisons.

The sourced circuit-type and prior event-rate features have not yet passed a one-family-at-a-time ablation against a frozen Stage 2 feature set. Keep them experimental until that comparison is recorded. High/low-speed car-track interactions remain blocked by missing measured car traits and circuit geometry.

## Data coverage decisions

- OpenF1 race-control data provides useful event labels from 2023 onward, but source labels are message-derived and do not provide a cutoff-time forecast archive.
- The local meeting API provides a coarse circuit type. Detailed track geometry linked from meeting metadata was unavailable to this ingestion, so no subjective corner-speed scores were invented.
- Historical weather forecasts at each race cutoff are absent. Observed weather remains excluded from pre-race model inputs.
- The current pace table lacks timestamped sector and speed-trap data, so measured car traits and car-track interactions cannot yet be tested responsibly.

## Promotion and reproduction

- Candidate event-risk report: [`../../ml/reports/SAFETY_CAR_EVAL.md`](../../ml/reports/SAFETY_CAR_EVAL.md)
- Data inventory: [`../../ml/reports/DATA_AUDIT.md`](../../ml/reports/DATA_AUDIT.md)
- Snapshot tooling: `ml/f1ml/lifecycle.py`, `ml/save_forecast_snapshot.py`
- Candidate tooling: `ml/train_candidate.py`
- Promotion is gated by saved chronological evidence, minimum race coverage, win log-loss improvement, Brier/calibration tolerances, and regulation-era reporting; rollback artifacts and promotion decisions are archived separately.
- The Stage 2 baseline remains the reference candidate. The existing 19-race 2026 diagnostic and this negative safety-car experiment do not justify promotion.
