# Runbook: race-weekend forecast runs (by hand)

**For:** Siddharth, until the runs are automated (see [`OPS_PLAN.md`](OPS_PLAN.md)).
**Written:** 2026-11-10 by workstream E. All times are **UTC**.

Every run writes a new, frozen snapshot to `ml/ledger/` (content-hashed, append-only; see
[`ml/ledger/README.md`](../ml/ledger/README.md)). Nothing is ever overwritten, so a run you
make by mistake is harmless: it is just one more snapshot. The Track Record page reads only
these snapshots.

## One-time setup

```bash
cd ml
```

The repo's `.venv` is not set up on this Mac; the Anaconda Python works (it is what the tests
and `.claude/launch.json` use). Check it once:

```bash
python -m pytest tests -q
```

## The four commands

| When | Command | What it does |
|---|---|---|
| Thursday | `python scripts/check_prices.py --season 2026 --round N --template` then edit `data/manual/fantasy_prices.csv`, then `python scripts/check_prices.py --season 2026 --round N` | Enter this round's F1 Fantasy prices (~15 min). Without valid prices the forecast still runs, but there is no Oracle team and the Fantasy Helper hides budget features. |
| **Before the lock** (deadline: lock minus 1 h) | `python race_weekend.py before --season 2026 --round N` | Downloads data, retrains (approved recipe; refit checks guard the swap), freezes the **pre-lock** snapshot. The output says `published >= 1 h before the team lock` when on time. |
| After qualifying, and again once the official grid is out | `python race_weekend.py update --season 2026 --round N` | Downloads quali/grid, freezes a **post-qualifying** snapshot with the models `before` saved. No retrain. |
| After the race (results usually within a few hours) | `python race_weekend.py after --season 2026 --round N` | Scores every snapshot of the race (winner, log-loss vs grid, fantasy points vs a last-3-races baseline), appends to `ledger/scores.jsonl`, adds the race to the calibration history. |

After each run:

```bash
python race_weekend.py verify
```

```bash
git add ml/ledger ml/models ml/reports ml/data && git commit -m "Weekend run: 2026 RN <stage>"
```

Pushing makes the snapshot public (git history is the second, independent timestamp). Push
when you're ready; that's your call.

If `before` prints `REFIT CHECKS FAILED`, the new models were saved in `models/candidates/`
for review and the forecast was made with the existing live models. The snapshot is still
valid. Tell the ML workstream; don't promote by hand.

## Remaining 2026 weekends

Session times are from Jolpica (`data/manual/weekend_schedule.csv`, refresh with
`python scripts/sync_schedule.py --season 2026`). **None of the three is a sprint weekend**
in the official schedule, so the lock is the start of qualifying.

### Do first: score Brazil (R20)

The Brazil forecast was saved by the old code (`reports/forecasts/2026_R20_*.csv`) and has
not been scored. Run this **before** the Las Vegas `before`, so the calibration history
includes it:

```bash
python race_weekend.py after --season 2026 --round 20
```

(Dry run on 2026-11-10: works, scores both legacy CSVs.)

### Las Vegas GP, round 21

Lock = qualifying start **Sat 21 Nov 04:00 UTC** (Fri 20 Nov, 20:00 in Las Vegas). Race Sun 22 Nov 04:00 UTC.

| Step | Run at (UTC) | Hard deadline | Command |
|---|---|---|---|
| Prices | Thu 19 Nov | Fri 20 Nov 18:00 | `python scripts/check_prices.py --season 2026 --round 21` |
| Pre-lock | **Fri 20 Nov 18:00** | **Sat 21 Nov 03:00** | `python race_weekend.py before --season 2026 --round 21` |
| Post-quali | Sat 21 Nov 08:00 | before Sun 22 Nov 04:00 | `python race_weekend.py update --season 2026 --round 21` |
| Grid set | Sat 21 Nov 22:00 (when the official grid is out) | Sun 22 Nov 03:30 | `python race_weekend.py update --season 2026 --round 21` |
| Score | Sun 22 Nov 12:00 | — | `python race_weekend.py after --season 2026 --round 21` |

### Qatar GP, round 22

Lock = qualifying start **Sat 28 Nov 18:00 UTC**. Race Sun 29 Nov 16:00 UTC. (MVP_SPEC guessed
Qatar would be a sprint weekend; the official schedule says standard. Check the game.)

| Step | Run at (UTC) | Hard deadline | Command |
|---|---|---|---|
| Prices | Thu 26 Nov | Fri 27 Nov 18:00 | `python scripts/check_prices.py --season 2026 --round 22` |
| Pre-lock | **Fri 27 Nov 18:00** | **Sat 28 Nov 17:00** | `python race_weekend.py before --season 2026 --round 22` |
| Post-quali | Sat 28 Nov 21:00 | before Sun 29 Nov 16:00 | `python race_weekend.py update --season 2026 --round 22` |
| Grid set | Sun 29 Nov 12:00 | Sun 29 Nov 15:30 | `python race_weekend.py update --season 2026 --round 22` |
| Score | Sun 29 Nov 21:00 | — | `python race_weekend.py after --season 2026 --round 22` |

### Abu Dhabi GP, round 23

Lock = qualifying start **Sat 5 Dec 14:00 UTC**. Race Sun 6 Dec 13:00 UTC.

| Step | Run at (UTC) | Hard deadline | Command |
|---|---|---|---|
| Prices | Thu 3 Dec | Fri 4 Dec 18:00 | `python scripts/check_prices.py --season 2026 --round 23` |
| Pre-lock | **Fri 4 Dec 18:00** | **Sat 5 Dec 13:00** | `python race_weekend.py before --season 2026 --round 23` |
| Post-quali | Sat 5 Dec 17:00 | before Sun 6 Dec 13:00 | `python race_weekend.py update --season 2026 --round 23` |
| Grid set | Sun 6 Dec 09:00 | Sun 6 Dec 12:30 | `python race_weekend.py update --season 2026 --round 23` |
| Score | Sun 6 Dec 18:00 | — | `python race_weekend.py after --season 2026 --round 23` |

## If something goes wrong

| Symptom | Do this |
|---|---|
| `... is not the next race in the data` | The ingest didn't pick up the previous race's result yet. Wait an hour and retry. |
| `already has a result; use after` | You're too late for a forecast; run `after`. |
| `published AFTER the pre-lock deadline` | The snapshot is kept and marked late (honest). The Track Record shows it as late. |
| `Ledger verification failed` | Someone edited a published file. Don't write anything; restore the file from git (`git checkout -- ml/ledger`) and run `verify` again. |
| A network error during ingest | Retry. Add `--skip-ingest` only if the data you already have includes what you need (the summary line prints `quali=… grid=…`). |
| Not sure what a run will do | Add `--dry-run`: it writes to a temporary ledger and leaves models, tracking and calibration files alone. |

## Dry run report: Las Vegas, 2026-11-10

Command: `python race_weekend.py before --season 2026 --round 21 --skip-ingest --dry-run`
(after a full ingest the same morning, which brought in the Brazil result).

- **Retrain:** pre_quali and post_quali on 112 races, approved recipe (pace on classified
  finishers; temperature from the 91-race out-of-sample history: 1.45 pre-quali, 0.86
  post-quali). post_sprint on 27 races (fallback temperature). **Refit checks passed.**
  Live models untouched (dry run). Runtime about 10 s.
- **Snapshot:** `2026-R21-pre_quali-…`, stage `pre_quali`, `published_before_lock: true`,
  `data_as_of` = Brazil race start, `feature_snapshot_id` and `sha256` recorded, scoring
  rules `2026.1`, 6,000 simulated weekends. Warnings recorded in the snapshot: qualifying
  points are projected from a simulated qualifying order; prices for R21 not entered, so no
  Oracle team.
- **Forecast (pre-qualifying, for the record only):** Antonelli 22% win, Leclerc 19%,
  Verstappen 17%, Norris 13%, Russell 12%. Projected fantasy points: Antonelli 23 (likely
  −10 to 38), Leclerc 22, Verstappen 20, Norris 19. Mercedes top constructor at 57 (27–79).
  The low end of the range is negative for everyone: it is the retirement case (−20).
- **Post-qualifying path:** exercised on Brazil (R20) in a dry run; stage `post_quali`.
- **`after`:** dry run on Brazil scores both legacy CSV forecasts.
- **Not tested:** the real ingest of Las Vegas qualifying (it hasn't happened), and a
  non-dry `before` (it would replace the live models; that's the first real run on 20 Nov).
