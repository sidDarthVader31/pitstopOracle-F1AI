# EXP-006: Stop double-counting retirements in the pace model

**Status:** done. **Decision: keep** "train the pace model on classified finishers only" (`F1ML_PACE_TRAIN=classified`) as the recommended recipe. The code default is unchanged (`all`) until the owner approves a change to the live model.
**Evaluation:** the same 91 walk-forward races as `ml/reports/EVAL.md`, with 2025–26 (43) and 2026 (19) shown separately, plus the internal market comparison on the 50 races in EXP-004. Calibration is today's 10-race temperature unless a row says otherwise.
**Branch:** `claude/ml-fixes-6f0fa2`. No model artifact was written or promoted.

**Adopted 2026-11-10** (owner approval): classified pace training is now the default for pre_quali and post_quali (post_sprint keeps all rows). See `docs/ML_LAUNCH_READINESS.md`, "Update 2026-11-10" for the regenerated `EVAL.md` numbers.

## Plain-language summary

The model has two parts. The **pace model** learns "how fast is this car and driver, relative to the field". The **DNF model** then randomly retires cars in each simulated race, at their team's retirement rate.

Until now the pace model learned from *every* result, retirements included. A driver who crashed out from 3rd is listed around 18th, so the pace model learned "this kind of car finishes 18th", and then the DNF model retired that car again on top. Unreliable cars were punished twice. Worse, a fast car's random crash made its pace look slower, which muddles the whole ranking. It's like judging a driver's race pace from a lap chart that includes the laps after the engine blew.

**The fix:** the pace model learns only from drivers who were classified at the finish. Reliability stays entirely with the DNF model.

**Result:** after qualifying, log-loss improves in every period. All 91 races: 1.443 → **1.305** (grid 1.488). 2025–26: 1.410 → **1.225** (grid 1.386). 2026: 1.637 → **1.249** (grid 1.617). Favourites stop being over-confident. In 2025–26, drivers priced at 50%+ were rated 66% and won 65%. Against the betting market the gap roughly halves, from +0.31 to +0.14, but the model still doesn't beat it.

## Hypothesis

Removing retirements from the pace model's training data (option A), or giving them much less weight (option B), removes the double count and improves post-quali log-loss without hurting pre-quali.

## The change

`ml/f1ml/ranking.py` `train_pairwise_ranker`, switch `F1ML_PACE_TRAIN`:

| Value | Pace-model training rows |
|---|---|
| `all` (default, today) | Every row with a numeric finishing position, retirements included (mean retiree position 18.8) |
| **`classified`** | Only rows with `dnf == 0`. Retirements are dropped from the pace model only. The DNF model still learns from them. |
| `dnf_weight:0.3` | Retirements kept, at 30% of their usual sample weight |

Only training rows change. Test races are still scored on the full entry list (the EXP-004 fix), and every simulated race still retires cars with the DNF model's probabilities.

## Results: after qualifying (baseline = official grid)

| Window | Variant | Log-loss | Grid | Model − today (95% CI) | Brier | Hit | Top-3 | Rank corr. | Finish MAE |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| All 91 | today (`all`) | 1.443 | 1.488 | — | 0.0328 | 47% | 85% | 0.659 | 3.22 |
| | **classified** | **1.305** | | −0.139 (−0.30 to +0.03) | 0.0309 | 54% | 89% | 0.676 | 3.11 |
| | dnf_weight 0.3 | 1.303 | | −0.140 (−0.28 to −0.00) | 0.0304 | 46% | 90% | 0.678 | 3.13 |
| 2022–24 (48) | today | 1.473 | 1.579 | — | 0.0312 | 52% | 81% | 0.659 | 3.21 |
| | **classified** | **1.376** | | −0.096 (−0.35 to +0.17) | 0.0315 | 48% | 88% | 0.688 | 3.03 |
| 2025–26 (43) | today | 1.410 | 1.386 | — | 0.0346 | 42% | 88% | 0.658 | 3.24 |
| | **classified** | **1.225** | | −0.186 (−0.39 to +0.03) | 0.0302 | 60% | 91% | 0.662 | 3.19 |
| | dnf_weight 0.3 | 1.234 | | −0.176 (−0.36 to +0.02) | 0.0303 | 47% | 93% | 0.666 | 3.19 |
| 2026 (19) | today | 1.637 | 1.617 | — | 0.0370 | 42% | 79% | 0.671 | 3.33 |
| | **classified** | **1.249** | | **−0.388 (−0.64 to −0.09)** | 0.0295 | 63% | 89% | 0.670 | 3.29 |
| | dnf_weight 0.3 | 1.276 | | −0.361 (−0.71 to +0.07) | 0.0276 | 42% | 89% | 0.673 | 3.31 |

Classified minus grid on 2025–26: −0.161 (95% CI −0.55 to +0.16). The model is better on average, but 43 races can't prove it.

**Calibration**, top bucket (50%+: n, said, happened), with today's calibration:

| Window | today (`all`) | classified |
|---|---|---|
| All 91 | 46: 69% / 59% | 51: 69% / 65% |
| 2025–26 | 23: 64% / 43% | **26: 66% / 65%** |
| Market window (50 races) | 23: 63% / 39% | 26: 65% / 62% |

**Internal market check** (post-quali winner, 50 races, market 1.172): today 1.485 (+0.31, CI +0.11 to +0.51, better in 16 races) → **classified 1.316 (+0.14, CI −0.01 to +0.31, better in 21 races)**. The model still loses on average. It's a benchmark only and never a model input.

## Results: before qualifying (baseline = prior form) and after the sprint

| Window | Variant | Pre-quali log-loss | Prior form | Model − today (95% CI) | Brier | Hit |
|---|---|---:|---:|---|---:|---:|
| All 91 | today | 1.803 | 2.403 | — | 0.0362 | 37% |
| | **classified** | **1.735** | | −0.069 (−0.22 to +0.08) | 0.0373 | 33% |
| | dnf_weight 0.3 | 1.804 | | +0.000 | 0.0362 | 35% |
| 2025–26 | today | 1.981 | 2.311 | — | 0.0400 | 30% |
| | **classified** | **1.838** | | −0.143 (−0.35 to +0.04) | 0.0392 | 21% |
| 2026 | today | 2.106 | 2.292 | — | 0.0392 | 21% |
| | **classified** | **1.932** | | −0.174 (−0.42 to +0.08) | 0.0376 | 21% |

Market before FP1 (48 races, market 1.735): today 1.989 (+0.25) → classified 1.825 (+0.09, CI −0.11 to +0.28).

**Post-sprint stage** (12 races, 2025–26): 1.369 → 1.509 (classified) and 1.650 (dnf_weight 0.3). Worse, on a tiny sample. The flag applies to every stage. The post-sprint model trains on only about 15–27 sprint weekends, and dropping retirements removes about 14% of those rows.

## Combined with EXP-005 calibration

EXP-005 was run on the old pace model. Re-running its calibration variants on top of this change (`python scripts/exp_calibration.py pace_classified`):

| Post-quali, classified pace model | All 91 | 2025–26 (− grid, 95% CI) | 2026 | Brier 25–26 | Top bucket 2025–26 | Market (50) |
|---|---:|---|---:|---:|---|---|
| today's 10-race temperature | 1.305 | 1.225 (−0.16, −0.55 to +0.16) | 1.249 | 0.0302 | 26: 66% / 65% | 1.316 (+0.14) |
| **rolling temperature** | **1.276** | **1.184 (−0.20, −0.62 to +0.10)** | **1.210** | 0.0294 | 17: 59% / 59% | **1.289 (+0.12, CI −0.02 to +0.26)** |
| rolling temperature + grid blend | 1.302 | 1.187 (**−0.20, −0.41 to −0.02**) | 1.310 | **0.0277** | 20: 59% / 65% | 1.348 (+0.18) |

With the double count gone, most of the over-confidence that EXP-005 attributed to calibration disappears too. The simple rolling temperature is then best on log-loss in every window. The grid blend is the only variant whose 2025–26 gain over the grid has a CI excluding zero, and it has the best Brier, but it's worse in 2026. **Recommended recipe: classified pace training + rolling temperature.** The grid blend remains the more conservative fallback.

## Limitations

- **Several variants were compared on the same races** (two here, plus the EXP-005 calibrators), so the best combination is somewhat flattering. The direction is consistent across 2022–24, 2025 and 2026, which makes chance less likely, but only the 2026 post-quali gain has a CI that excludes zero.
- Pre-quali: log-loss improves, but the top-pick hit rate drops (37% → 33% overall; 30% → 21% in 2025–26) and the all-window Brier is slightly worse (0.0362 → 0.0373).
- The post-sprint stage gets worse (12 races). If adopted, keep `all` for `post_sprint`, or retrain it, and re-check.
- Treating retirements as properly *censored* ("he was at least as good as where he retired") would use more information, but needs lap-by-lap running positions we don't store. Not tried.
- Retirement labels come from the Jolpica status text (`features._dnf_labels`). Classified finishers listed as "+1 Lap" etc. are treated as finishers, which is correct.

## Decision

**Keep `classified`** as the recommended pace-model training set for pre-quali and post-quali, together with the EXP-005 calibration as a rolling temperature. **Owner decision needed** before switching the default in `ranking.py` and retraining the live models. If approved, wire both changes into `race_weekend.train_weekend_bundle` and the walk-forward together, then regenerate `EVAL.md`.

## Reproduce

```bash
cd ml
F1ML_PACE_TRAIN=classified     python scripts/wf_dump.py pace_classified   # ~6 min each
F1ML_PACE_TRAIN=dnf_weight:0.3 python scripts/wf_dump.py pace_dnfw03
python scripts/exp_calibration.py pace_classified                          # calibration + market check
# per-race metrics: data/cache/experiments/<tag>_metrics.json (git-ignored)
```
