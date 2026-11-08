# EXP-001 — Cutoff-safe baseline

**Status:** evaluated; no candidate promoted
**Scope:** Roadmap Phase A / companion Phase 1 forecast-integrity baseline
**Evaluation:** expanding walk-forward over the 19 completed 2026 races, with each race held out as a full field and only earlier races available for fitting/calibration.

## Question

Can the existing HistGradientBoosting pace and DNF models be evaluated on cutoff-safe inputs with corrected labels and an inference-matched race scorer, without changing the model family or promoting a candidate?

## Changes

- Removed current-weekend race pace, practice, observed weather and sprint-result fields from both default model feature views. Historic snapshots do not establish when these values were available.
- Corrected classified-status handling for `Lapped`, retained the full labeled DNF population, and separated ranking rows from DNF rows.
- Shifted circuit history at the race level, corrected practice long-run fallback precedence, grouped constructor rating updates by event, and cleared stale previous-event weekend inputs when building upcoming predictions.
- Restricted evaluation to chronological expanding folds, calibrated from the ten preceding races, and used the production Plackett–Luce scorer path for evaluation. Each fold used 1,200 Monte Carlo draws.
- Made `train_and_eval` report-only: it writes metrics and charts but neither writes nor promotes model artifacts.

## Results

The test set contains 19 races from 2026-03-08 through 2026-11-01. These metrics are a small new-regulation-era diagnostic, not a stable estimate of future performance.

| Stage / baseline | Win hit rate | Top-3 hit rate | Win log loss | Win Brier | Rank correlation | Finish MAE |
|---|---:|---:|---:|---:|---:|---:|
| Pre-quali model | 15.8% | 63.2% | 2.136 | 0.0417 | 0.698 | 3.439 |
| Prior-form baseline (pre) | — | — | 3.135 | — | — | — |
| Post-quali model | 42.1% | 78.9% | 3.437 | 0.0374 | 0.771 | 2.939 |
| Official-grid baseline | 68.4% | — | 1.617 | — | — | — |
| Qualifying-gap baseline | — | — | 1.576 | — | — | — |

An additional transfer diagnostic trained only before 2026 and tested on 2026 had post-quali win hit rate 52.6% and win log loss 1.557. This is a separate diagnostic, not the expanding-window result or a promotion decision.

## Decision

Keep the data-contract and leakage corrections. The pre-quali model improves on the included prior-form baseline by win log loss, but the 19-race sample is too small to establish robust value. The post-quali model does not beat the official-grid or qualifying-gap baseline on win log loss. Do not promote a model artifact. Investigate probability calibration and compare simpler pace models under the same frozen walk-forward protocol in Phase B.

## Reproduction and artifacts

- Full report: [`../../ml/reports/EVAL.md`](../../ml/reports/EVAL.md)
- Machine-readable metrics: [`../../ml/reports/metrics.json`](../../ml/reports/metrics.json)
- Implementation: `ml/f1ml/{specs,features,ratings,predict,ranking,train,eval}.py`
- Tests: `cd ml && .venv/bin/python -m pytest tests`
- `model_artifacts_written` is `false`; no champion/candidate artifact was written or promoted.

## Limitations

- Only 19 evaluation races are available in 2026; results have high sampling uncertainty.
- Historical forecast snapshots for weather and practice are unavailable, so those current-event feature families remain excluded.
- At the time of EXP-001, sprint-cutoff-specific forecasts and explicit DNF simulation were not implemented; they were added in EXP-002.
- `circuit_grid_to_finish_delta` is sparse (1,505 null driver-race rows) and relies on fit-time imputation.
- The DNF label is a deterministic classified/non-classified proxy from official status strings; cause-specific risk is not modeled.
