# EXP-008: Calibrate the chance of finishing in the points

**Status:** done. **Decision: recommend** the "ramp" sampler (a softer midfield) for pre-quali and post-quali. Owner approval needed. It goes live **after** the 2026 shadow races (Abu Dhabi, 6 Dec); until then the live recipe stays frozen.
**Evaluation:** the walk-forward from `scripts/wf_dump.py adopted`: the adopted recipe (EXP-005 + EXP-006), 92 races from 2022 R21 to 2026 R20, the same folds as `ml/reports/EVAL.md`. 2025–26 (44 races) and 2026 (20 races) are shown separately.
**Branch:** `claude/g3-model-work`. No model artifact written or promoted; `ml/models/` untouched.

## Plain-language summary

The model's chance that a driver "scores points" (finishes in the top 10) was badly calibrated. After qualifying it gave 238 driver-races less than 1%, and 8 of them scored points. Drivers priced at 1–5% scored 13% of the time. In 2025–26 it was no better than a simple lookup table of "how often does P-whatever on the grid score points".

**The cause** (found in EXP-007): one "confidence dial" (the temperature) controls how predictable the *whole* finishing order is. That dial is tuned to pick the **winner**. The fight for P1 is fairly predictable, but the scrap for P7–P12 is much more random: strategy, safety cars, first-lap chaos. With the winner's dial, the model thought the midfield order was nearly as settled as the front.

**The fix:** the simulation now picks the finishing order one position at a time. The winner is still picked with the winner's dial, so **win chances don't change at all**. Each later position is picked with a softer dial, rising from 1× at P1 to *m*× at P10 and flat after that. Think of it as P1 being decided on pace, and the midfield also decided by luck. *m* is fitted only on earlier races; it settles at 3–4×.

**Result after qualifying:** points log-loss falls from 0.505 to **0.459** (all races), 0.539 → **0.480** (2025–26) and 0.503 → **0.461** (2026). It now beats the grid-slot lookup table in every window, with the 95% interval excluding zero. Every calibration bucket with a real sample is within about 2 points of what happened. Podium chances also improve slightly, and expected finishing position doesn't change.

## Hypothesis

Points chances are over-confident because the winner's temperature is applied to every position. A position-dependent temperature, fitted on earlier races, will make points chances calibrated and better than the grid-slot table, without hurting win, podium or expected-finish quality.

## The change

`ml/f1ml/calibration.py` (new, tested, **not used by the live run yet**):
- `sequential_pl_ranks(scores, retired, temperature, rest_multiplier, schedule)` draws the classified order position by position. Position *k* uses temperature × multiplier(*k*), and retirements come from the same DNF draws as today. With multiplier 1 it is exactly today's Plackett–Luce model.
- `position_multipliers(n, m, schedule)`: `step` (1 for P1, *m* for P2 onwards) or `ramp` (1 → *m* linearly to P10, flat after).
- `fit_rest_multiplier(losses)` picks *m* from {1, 1.25, 1.5, 2, 2.5, 3, 4, 5} with the lowest points log-loss over **earlier** races. It needs at least 10 races, otherwise *m* = 1.

Experiment: `ml/scripts/exp_points.py`. Each test race uses that fold's own pace scores, DNF chances and winner temperature, so only the points/podium step changes. 2,000 draws per race.

| Variant | What it does |
|---|---|
| `adopted` | Today: one temperature for the whole order |
| `step` | P1 at the fitted temperature, P2 onwards × *m* |
| **`ramp`** | **Multiplier rising from 1 (P1) to *m* (P10)** |
| `rank_blend` | Logit blend of today's points chance with the grid-slot / form-rank table, weight fitted |
| `platt` | Logistic recalibration of today's points chance |
| `rank_rate` (baseline) | Historical points rate for the driver's grid slot (post-quali) or prior-form rank (pre-quali), from earlier races |

## Results: after qualifying (baseline: grid-slot table)

Points log-loss (lower is better), with model minus baseline and its 95% interval:

| Variant | All 92 | 2025–26 (44) | 2026 (20) | Points Brier 25–26 | Podium log-loss 25–26 | Under 1% (n / scored), all |
|---|---|---|---|---:|---:|---|
| grid-slot table | 0.507 | 0.508 | 0.514 | 0.1659 | 0.213 | 0 / 0 |
| adopted (today) | 0.505 (−0.001, −0.04 to +0.04) | 0.539 (+0.031, −0.02 to +0.08) | 0.503 | 0.1668 | 0.213 | **238 / 8** |
| step | 0.459 (−0.048) | 0.479 (−0.029) | 0.459 | 0.1553 | 0.209 | 25 / 2 |
| **ramp** | **0.459 (−0.047, −0.07 to −0.03)** | **0.480 (−0.028, −0.05 to −0.01)** | **0.461 (−0.054, −0.10 to −0.02)** | **0.1558** | **0.199** | 16 / 2 |
| rank_blend | 0.462 (−0.045) | 0.487 (−0.021, −0.04 to +0.01) | 0.468 | 0.1578 | 0.213 | 58 / 3 |
| platt | 0.472 (−0.035) | 0.497 (−0.011) | 0.471 | 0.1611 | 0.213 | 87 / 3 |

Calibration of points chances, all 92 races (n / said / happened):

| | 0–1% | 1–5% | 5–20% | 20–50% | 50–80% | 80–95% |
|---|---|---|---|---|---|---|
| adopted | 238 / 0% / 3% | 194 / 3% / **13%** | 274 / 11% / **24%** | 174 / 33% / 33% | 236 / 69% / **54%** | 756 / 88% / 84% |
| **ramp** | 16 / 1% / 12% | 102 / 3% / 2% | 427 / 12% / 12% | 422 / 32% / 34% | 296 / 66% / 65% | 609 / 88% / 87% |

Ramp vs today, paired 95% intervals: points log-loss −0.046 (−0.07 to −0.02) over all races; podium log-loss **−0.018 (−0.03 to −0.01)**; expected-finish error −0.01 places (−0.04 to +0.03), i.e. no change. Win chances are identical by construction (tested).

## Results: before qualifying (baseline: form-rank table)

| Variant | All 92 | 2025–26 | 2026 | Podium log-loss all / 25–26 |
|---|---|---|---|---|
| form-rank table | 0.547 | 0.532 | 0.510 | 0.296 / 0.257 |
| adopted (today) | 0.552 | 0.567 | 0.516 | 0.296 / 0.257 |
| **ramp** | **0.505 (−0.042, −0.07 to −0.02)** | 0.522 (−0.011, −0.04 to +0.02) | 0.496 (−0.014) | **0.279 / 0.248** |
| rank_blend | 0.504 | 0.515 | 0.482 | 0.296 / 0.257 |

Before qualifying, ramp beats the form-rank table over all races, and on the point estimate in 2025–26 and 2026 (the intervals include zero there). The calibration buckets are within about 3 points apart from the two drivers under 1%. `rank_blend` is slightly better on points alone before qualifying, but doesn't improve podium chances and isn't a single simulation. Fantasy points need one consistent set of simulated races (`f1ml.fantasy_sim`), so ramp is the cleaner choice.

The fitted multiplier settles quickly: 2–2.5 in the first eligible races, then 3–4 from 2024 onwards. The latest fit (2026 R20) is 4 in both stages.

## Limitations

- **Chosen after looking.** Five variants on the same races; ramp and step are almost tied. The size of the gain over today (about 0.05) is far larger than the gap between variants, so the choice of variant matters little.
- The points rule here is "top 10 in the race". Sprint points and DNF penalties enter fantasy points separately.
- **Not wired in.** Live wiring needs (1) points outcomes stored in `reports/oos_calibration_history.parquet` (today it stores only the winner), so *m* can be refitted each weekend, and (2) `ranking.simulate_race_draws` / `fantasy_sim` to use `sequential_pl_ranks`. About a day's work, and it must go through `train_stage_bundle` so the evaluation and the live run stay identical.

## Decision

**Recommend `ramp`** for pre-quali and post-quali points and podium chances, and for the fantasy points built from the same draws. Owner approval needed; wire it in after 6 Dec. Once wired, the "calibrated chance of scoring points" claim can move to **yes** (post-quali) and **yes, with a caveat** (pre-quali).

## Reproduce

```bash
cd ml
python scripts/wf_dump.py adopted      # ~5-8 min, git-ignored cache
python scripts/exp_points.py adopted   # writes ml/reports/exp008_points_adopted.json
```
