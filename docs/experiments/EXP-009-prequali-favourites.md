# EXP-009: Pre-qualifying favourites

**Status:** done. **Decision: small optional improvement.** A recency-weighted temperature for pre-quali only (half-life 12 races) helps a little in every window, but the gain is within the noise. Keep the MVP rule that pre-quali chances show as bands, with no band above 50%. Owner decision 2026-11-14: not adopted.
**Evaluation:** the same 92-race walk-forward as EXP-008 (adopted recipe). Baseline: the prior-form forecast. The internal market check (Polymarket before the weekend, 48 races) is a benchmark only, never a model input.
**Branch:** `claude/g3-model-work`. No model artifact written or promoted.

## Plain-language summary

Before qualifying, the model's favourites looked too confident in recent seasons:
- In 2025–26, drivers priced at 35% or more won 5 of 21 times (about 24%), against roughly 45% forecast.
- Outsiders priced at 5–10% won about twice as often as forecast (16% vs 7%).

Over all 92 races the picture is fine: drivers priced at 50% or more said 69% and won 59% (n = 27), within the ±10-point target. So the problem is recent races, and it's small in absolute terms: about 20 driver-races.

**Cause:** the confidence dial is fitted on every earlier race, back to 2022. That includes the 2022–23 Red Bull era, when the pre-race favourite almost always won. Those years pull the dial towards "very sure", while 2025–26 have been much more open.

**What we tried:**
1. Blending towards an equal field.
2. Blending towards the prior-form forecast.
3. Fitting the dial with more weight on recent races.

Blending doesn't help: the fit picks a blend weight of zero. Recency weighting helps a little everywhere: 2025–26 log-loss 1.884 → 1.860, 2026 1.968 → 1.954, and the favourite's average chance comes down from 37% to 33% (its actual win rate is 27%). But every difference is inside the noise.

## Results

Win log-loss (lower is better). The baseline is prior form: 2.406 (all), 2.318 (2025–26), 2.309 (2026). The model beats it in every window by 0.34–0.73, and the confidence intervals exclude zero.

| Variant | All 92 | 2025–26 (44) | 2026 (20) | Favourite said / won, 2025–26 | 50%+ bucket, all (n / said / won) | Market check (model vs market 1.753) |
|---|---:|---:|---:|---|---|---|
| adopted (rolling temperature, all history) | 1.677 | 1.884 | 1.968 | 37% / 27% | 27 / 69% / 59% | 1.879 |
| equal-field blend | 1.688 | 1.899 | 1.966 | 37% / 27% | 29 / 66% / 59% | 1.895 |
| prior-form blend | 1.677 | 1.885 | 1.966 | 37% / 27% | 26 / 70% / 58% | 1.881 |
| last 24 races only | 1.677 | 1.855 | 1.956 | 32% / 27% | 22 / 74% / 68% | 1.854 |
| **recency-weighted, half-life 12 races** | **1.667** | **1.860** | **1.954** | **33% / 27%** | 23 / 73% / 70% | **1.857** |
| recency-weighted, half-life 24 | 1.670 | 1.867 | 1.957 | 34% / 27% | 23 / 72% / 65% | 1.865 |

Half-life 12 minus adopted (paired 95% interval): all −0.010 (−0.04 to +0.02), 2025–26 −0.024 (−0.07 to +0.02), 2026 −0.015 (−0.06 to +0.03).

The same weighting after qualifying is slightly **worse** (2025–26 +0.013, 2026 +0.015, intervals include zero). So it would apply to pre-quali only.

2025–26 calibration with half-life 12 (n / said / won): 5–10% bucket 70 / 7% / 13%; 35–50% bucket 15 / 42% / 33%. Still a touch top-heavy, but less so (it was 7% / 16% and 43% / 29%).

## Limitations

- Six variants compared on the same races, so selection optimism is likely. The gain is about the size of that optimism.
- Tiny top buckets: in 2025–26 only 4 driver-races were priced at 50% or more before qualifying (none won).
- The next regulation change will repeat this problem. Recency weighting is the general answer, which is a reason to prefer it.

## Decision

- **Product (already the MVP rule): keep it.** Before qualifying, show chance bands with no band above 50%. No "calibrated pre-quali win chance" claim.
- **Model: optional.** Adopt the recency-weighted temperature (half-life 12) for pre-quali only, after 6 Dec, if the owner wants the small gain. It's a one-line change inside `calibration.fit_rolling_temperature` (add race weights), applied only to `pre_quali`. The live run and the walk-forward share it automatically through `train_stage_bundle`.

## Reproduce

```bash
cd ml
python scripts/wf_dump.py adopted        # once
python scripts/exp_prequali.py adopted   # writes ml/reports/exp009_prequali_adopted.json
```
