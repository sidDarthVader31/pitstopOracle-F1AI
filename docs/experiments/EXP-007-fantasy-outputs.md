# EXP-007: Are the model's full-race outputs useful for F1 Fantasy?

**Status:** done. **Decision: keep** the fix to `fantasy.py` (expected points now use the simulated finishing distribution). **Report:** the model is genuinely useful for **expected finishing position**, and slightly for **pre-quali podium chances**. It is **not** better than simple baselines for **points-finish (top-10) chances** or for **picking a 5-driver team**.
**Evaluation:** the same 91 walk-forward races as `ml/reports/EVAL.md`, scored for pre-quali (when Fantasy teams lock) and post-quali. 2025–26 (43 races) and 2026 (19 races) are shown separately. Model = the approved recipe unless marked "EXP-006".
**Branch:** `claude/ml-fixes-6f0fa2`. No model artifact was written or promoted.

## Plain-language summary

Fantasy players don't only care who wins. They care whether each of their five drivers scores points, finishes on the podium, and gains places. They also pick their team **before qualifying**. So we scored the model's whole-race forecast, not just the winner.

- **Where the model helps:** its **expected finishing position** is closer to reality than a simple baseline: about 0.2–0.45 places per driver better, every period, both before and after qualifying. Before qualifying, its **podium chances** are a little better than the prior-form baseline.
- **Where it doesn't:** its **"will he score points?"** chance is no better than a lookup table of "how often has a driver starting from this grid slot (or this form rank) scored points". After qualifying it's worse. The cause is technical: the confidence setting is tuned to predict the *winner*, which leaves the midfield order too predictable. In 2025–26 the model gave 75 drivers under 1% to score points after qualifying, and 4 of them did.
- **Bug fixed:** the Fantasy expected-points calculation ignored the simulation and spread every driver's chances over all 20 positions. A clear favourite kept roughly a 1-in-5 chance of finishing outside the points. It now uses the simulated distribution, which cuts the post-quali points error by about 0.9 Fantasy points per driver.
- **Team picking:** choosing the 5 drivers with the most expected points scores about the same as "pick the top 5 on the grid" (post-quali) or "the top 5 by recent form" (pre-quali). There's no edge yet.

## Questions

1. How good are the model's points-finish and podium probabilities, expected position and finishing order, compared with simple baselines, before and after qualifying?
2. Does `fantasy.py` use the model's finishing distribution correctly?
3. Where is the model genuinely useful for Fantasy players?

## Method

From each walk-forward test race's outputs (`scripts/wf_dump.py`), `scripts/exp_fantasy.py` scores:

| Metric | What it measures (lower is better unless noted) |
|---|---|
| Points / podium log-loss and Brier | Per driver yes/no: did he finish top 10 (`points_finish`) or top 3 (`podium`)? Disqualified drivers and non-starters count as "no". |
| Finish MAE | Expected finishing position vs actual, for classified finishers |
| Rank correlation (higher is better) | Spearman correlation between the predicted order and the actual result |
| xP MAE, xP rank correlation | Expected Fantasy **race** points vs actual (finish points, the DNF penalty, and post-quali the quali bonus and places gained or lost; sprint excluded) |
| Pick-5 points (higher is better) | Actual race points of the 5 drivers with the highest expected points (no budget) |

**Baselines**, each fitted only on races before the test race:
- `rank_rate`: the historical points (or podium) rate for each **baseline rank**. That's the grid slot after qualifying and the prior-form rating rank (`car_driver_combo`) before it. "P7 on the grid scores points x% of the time."
- `rank_mean`: the historical average finishing position for each baseline rank. `baseline_rank` is the rank itself.
- `base_rate`: everyone gets 10/n (points) or 3/n (podium).
- For team picking: the top 5 by baseline rank. "Hindsight best" is the ceiling.

## Results: before qualifying (when teams lock)

| Metric | Model | Simple baseline | Model − baseline (95% CI) | EXP-006 model |
|---|---:|---:|---|---:|
| **2025–26 (43 races)** | | | | |
| Points log-loss | 0.557 | rank_rate 0.537 (base rate 0.691) | +0.021 (−0.02 to +0.06) | 0.562 |
| Points Brier | 0.183 | 0.179 | +0.005 (−0.01 to +0.02) | 0.184 |
| Podium log-loss | **0.258** | 0.271 | −0.014 (−0.04 to +0.02) | **0.249** (−0.022, −0.04 to +0.00) |
| Finish MAE (places) | **2.77** | rank_mean 2.96 (raw rank 3.23) | **−0.18 (−0.34 to −0.01)** | **2.75** |
| Rank correlation | 0.587 | prior-form rank 0.623 | −0.037 (−0.09 to +0.01) | 0.582 |
| **2026 (19 races)** | | | | |
| Points log-loss | 0.536 | 0.519 | +0.017 (−0.03 to +0.07) | 0.541 |
| Podium log-loss | **0.254** | 0.275 | −0.021 (−0.05 to +0.01) | **0.252** |
| Finish MAE | **2.62** | 2.89 | **−0.27 (−0.45 to −0.09)** | **2.56** |
| Rank correlation | 0.610 | 0.671 | **−0.061 (−0.12 to −0.01)** | 0.616 |
| **All 91** | | | | |
| Points log-loss | 0.549 | 0.549 | −0.000 (−0.03 to +0.04) | 0.556 |
| Podium log-loss | 0.313 | 0.310 | +0.003 | 0.294 |
| Finish MAE | **2.76** | 3.15 | **−0.38 (−0.53 to −0.24)** | **2.69** |

## Results: after qualifying

| Metric | Model | Simple baseline (grid) | Model − baseline (95% CI) | EXP-006 model |
|---|---:|---:|---|---:|
| **2025–26** | | | | |
| Points log-loss | 0.547 | rank_rate **0.505** | +0.042 (−0.00 to +0.09) | 0.604 (+0.099, +0.03 to +0.17) |
| Points Brier | 0.171 | 0.165 | +0.006 (−0.00 to +0.02) | 0.176 |
| Podium log-loss | 0.231 | **0.210** | +0.020 (−0.01 to +0.05) | 0.239 |
| Finish MAE | **2.30** | rank_mean 2.63 (raw grid 2.81) | **−0.33 (−0.48 to −0.19)** | **2.27** |
| Rank correlation | 0.658 | 0.654 | +0.004 (−0.02 to +0.03) | 0.662 |
| **2026** | | | | |
| Points log-loss | 0.546 | **0.509** | +0.037 (−0.03 to +0.13) | 0.636 |
| Podium log-loss | 0.232 | **0.210** | +0.022 | 0.217 |
| Finish MAE | **2.18** | 2.61 | **−0.44 (−0.68 to −0.21)** | **2.17** |
| **All 91** | | | | |
| Points log-loss | 0.514 | 0.506 | +0.009 (−0.02 to +0.04) | 0.538 |
| Podium log-loss | 0.254 | **0.231** | **+0.023 (+0.00 to +0.04)** | 0.264 |
| Finish MAE | **2.39** | 2.65 | **−0.26 (−0.37 to −0.16)** | **2.28** |

**Why the points chances fall short.** One Plackett–Luce "temperature" sets how predictable the whole finishing order is. It is fitted to the **winner** only. Diagnostic on 2025–26 post-quali, tuned on the test races themselves, so not a fair score:

| Temperature × | 1 (today) | 1.5 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Points log-loss | 0.540 | 0.518 | 0.510 | **0.509** | 0.517 |
| Podium log-loss | 0.224 | **0.221** | 0.225 | 0.243 | 0.262 |
| Winner log-loss | **1.409** | 1.430 | 1.513 | 1.706 | 1.874 |

The midfield needs a 2–3× softer setting than the front. Even then the points log-loss only *matches* the grid-slot table (0.505). The EXP-006 model is sharper still, so its points chances are worse after qualifying (115 drivers under 1%, 11 scored) even though its winner and finishing-position forecasts are better.

## Fantasy expected points (`ml/f1ml/fantasy.py`)

**Problem found.** `expected_driver_fantasy_points` ignored the simulated finishing distribution (`finish_p{k}_prob`). It used only `expected_finish` and spread weight `1/(1 + |position − expected finish|)` over **every** position. A driver expected to finish 1.5 kept about 19% on finishing outside the points (and only 18% on winning), so favourites were under-rated and backmarkers over-rated.

**Fix.** Use the Monte Carlo distribution × (1 − DNF probability), plus DNF probability × the DNF score. Fall back to the old approximation only when no distribution is present. A grid slot of 0 is no longer read as "started P0". Tests: `ml/tests/test_fantasy.py`.

| | Old formula | Fixed | Fixed − old (95% CI) |
|---|---:|---:|---|
| Post-quali xP MAE, 2025–26 (Fantasy pts/driver) | 8.20 | **7.28** | **−0.93 (−1.14 to −0.69)** |
| Post-quali xP rank correlation, 2025–26 | 0.564 | **0.593** | +0.029 (+0.01 to +0.04) |
| Post-quali xP rank correlation, 2026 | 0.513 | **0.570** | +0.057 (+0.04 to +0.08) |
| Pre-quali xP rank correlation, 2025–26 | 0.412 | **0.473** | +0.061 (+0.05 to +0.07) |
| Pre-quali pick-5 points, 2025–26 / 2026 | 52.9 / 44.1 | **56.2 / 51.2** | +3.3 (+0.6 to +6.3) / +7.1 (+2.5 to +12.4) |
| Pre-quali xP MAE, 2025–26 | **6.74** | 7.02 | +0.28 (+0.13 to +0.43) |

The fixed version ranks drivers better (what team-picking needs) everywhere. Pre-quali its absolute points error is slightly larger, because the old flat formula shrank everyone towards the middle.

**Team picking vs simple baselines** (actual race points of the top 5 by expected points; hindsight ceiling about 80 pre-quali and 122 post-quali):

| | Model (fixed xP) | Top 5 by baseline rank | Model − baseline (95% CI) |
|---|---:|---:|---|
| Pre-quali, 2025–26 | 56.2 | 56.1 (prior form) | +0.1 (−5.5 to +5.4) |
| Pre-quali, 2026 | 51.2 | 49.8 | +1.3 (−6.5 to +9.1) |
| Post-quali, 2025–26 | 97.1 | 101.0 (grid) | −3.8 (−8.3 to +0.4) |

**Pre-quali expected points miss the qualifying part of the score.** Before qualifying the code has no quali or grid position, so the quali bonus (up to 10 points) and places gained or lost count as zero. Fantasy teams lock at exactly this point, so pre-quali xP understates front-runners and treats every driver as starting where he finishes. A pre-quali **qualifying-position forecast** would fill this gap (future work, listed below).

## Where the model is genuinely useful for Fantasy

| Output | Before qualifying | After qualifying | Product guidance |
|---|---|---|---|
| Expected finishing position | **Better** than the rank table (−0.2 to −0.4 places, CI excludes 0 in every window) | **Better** (−0.26 to −0.44) | Safe to feature: "expected finish" |
| Podium chance | Slightly better (2025–26 and 2026, CI touches 0) | Worse than the grid-slot table | Show pre-quali; after quali, don't claim more than "grid-based" |
| Points-finish chance | Same as the prior-form table | Worse than the grid-slot table | Show with care. Needs its own calibration (follow-up below) |
| Finishing order (rank correlation) | Slightly worse than prior form in 2026 | Same as the grid | No claim |
| Expected Fantasy points / team pick | Same as top-5 by form | Same as or slightly below top-5 by grid | Useful as a transparent explainer, not as an edge |
| Winner | Better than prior form (EXP-004/005/006) | About level with or better than the grid after EXP-005/006 | Loses to the market (internal benchmark only) |

The internal market benchmark has only the winner and podium markets (EXP-004), so no market comparison exists for points finishes or Fantasy points.

## Limitations

- Sprint points, fastest lap, constructor scores and price/budget constraints are excluded. Pick-5 ignores the budget, so it measures ranking quality, not a real lineup.
- `DNF_PENALTY = -15` and the scoring tables in `fantasy.py` must be checked against the 2027 F1 Fantasy rules before launch.
- The baselines are fitted on all earlier races since 2022. Grid-slot points rates drift a little between eras.
- 43 races (2025–26) and 19 (2026). Most per-metric CIs are a few hundredths wide and overlap.

## Decision

- **Keep** the `fantasy.py` fix (distribution-based expected points), committed with tests. It applies to the live dashboard right away, through `predict.weekend_card`. It doesn't change any model.
- **Report** to the product (workstream C/E): lead with expected finish and pre-quali podium chances. Don't claim points-finish or team-pick accuracy beyond the grid or form.
- **Proposed next experiments** (one change each):
  1. A separate temperature for positions beyond the winner (or fit the temperature on the full finishing-order likelihood), aimed at points-finish log-loss ≤ the grid-slot table.
  2. A pre-quali qualifying-position forecast, so pre-quali xP includes the quali bonus and places gained.

## Reproduce

```bash
cd ml
python scripts/wf_dump.py baseline       # if not cached (~7.5 min)
python scripts/exp_fantasy.py baseline   # ~6 s; also: pace_classified
# writes ml/reports/exp007_fantasy_<tag>.json
```
