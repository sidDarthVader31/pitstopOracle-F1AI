# EXP-010: The post-sprint stage

**Status:** done. **Decision: retire the post-sprint stage (approved by the owner 2026-11-14; wire in after 6 Dec).** On sprint weekends, once GP qualifying is known, use the adopted post-quali model. No sprint weekend remains in 2026, so the shadow races aren't affected either way; the change can go in before the 2027 season.
**Evaluation:** every sprint weekend in the walk-forward where the post-sprint stage can be trained: 12 weekends, 2025 R2 to 2026 R17 (6 in 2026). Same folds as `ml/reports/EVAL.md`. All variants are scored on the same drivers.
**Branch:** `claude/g3-model-work`. No model artifact written or promoted.

## Plain-language summary

On sprint weekends the model has a third stage, used after the Saturday sprint and GP qualifying, which adds the sprint result as an input. Since the walk-forward started refitting it the way the live run does, it scores **1.524**, worse than the plain starting grid (1.266).

We compared it against simply **ignoring the sprint** and using the normal after-qualifying model. That model scores **1.178** on the same weekends. It beats the post-sprint stage and the grid, and picks the winner in 6 of 12 (the post-sprint stage managed 3).

**Why:** the post-sprint model can only learn from sprint weekends, about 15–26 of them in each fold. The post-quali model learns from every race (70–110). The sprint result adds little that qualifying hasn't already shown: in F1 terms, the sprint mostly confirms Saturday's pace. Training on a small, specialised dataset loses more than the extra input gains. Applying the EXP-006 fix (train on classified finishers) to the post-sprint model makes it worse still (1.629).

## Results

Win log-loss (lower is better) on the same drivers of the same sprint weekends.

| Variant | 12 weekends (2025–26) | 6 weekends (2026) | Hit | Favourite said / won | Minus post-quali (95% interval) |
|---|---:|---:|---:|---|---|
| post-sprint (today) | 1.524 | 1.784 | 3/12 | 38% / 25% | +0.35 (−0.26 to +0.87) |
| **post-quali model (ignore the sprint)** | **1.178** | **1.176** | **6/12** | **55% / 50%** | — |
| post-sprint, classified pace + post-quali temperature | 1.629 | 1.682 | 5/12 | 53% / 42% | **+0.45 (+0.01 to +0.99)** |
| 50/50 mix of post-quali and post-sprint | 1.187 | 1.328 | 6/12 | 44% / 50% | +0.01 (−0.28 to +0.26) |
| grid baseline | 1.266 | 1.099 | 7/12 | 39% / 58% | +0.09 (−0.36 to +0.46) |

In 2026 alone (6 weekends) the grid edges the post-quali model, 1.099 vs 1.176. The interval is very wide (−0.77 to +0.37).

## Limitations

- **Twelve weekends is a tiny sample.** Only the classified variant is clearly worse than the post-quali model. The post-quali model's advantage over today's post-sprint stage (−0.35) is large but not statistically proven.
- Retiring the stage means the sprint result isn't used at all. A better use of it is possible later, for example a sprint-vs-quali adjustment trained on every race. That isn't worth building before launch.
- Fantasy scoring on sprint weekends still needs sprint points. Those come from the separate sprint simulation in `fantasy_sim` and don't depend on this stage.

## Decision

**Retire `post_sprint` as a forecast stage.** On sprint weekends:
- before GP qualifying, use `pre_quali` (already the rule since workstream D's fix);
- after GP qualifying, use `post_quali`, whether or not the sprint has run.

Code change after approval:
- `race_weekend._stage_for` and `predict._race_status_row` / `race_status` stop choosing `post_sprint`;
- the weekly retrain stops training it;
- `EVAL.md` reports post-quali on sprint weekends instead.

That's a small, testable change. It must land before the first 2027 sprint weekend.

## Reproduce

```bash
cd ml
python scripts/wf_dump.py adopted      # once
python scripts/exp_sprint.py adopted   # writes ml/reports/exp010_sprint_adopted.json
```
