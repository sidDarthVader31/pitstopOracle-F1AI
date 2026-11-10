# Forecast ledger (public track record)

Every forecast Pitstop Oracle publishes is frozen here **before** the session it predicts,
and is never edited afterwards. Written only by `race_weekend.py` (`before` / `update`);
scored by `race_weekend.py after`.

| Path | What |
|---|---|
| `snapshots/<season>/R<round>/<stage>__<generated_at>__<sha12>.json` | One frozen forecast. The last 12 characters of the name are the start of the file's SHA-256. |
| `index.jsonl` | One line per snapshot: id, stage, `generated_at`, lock time, model version, path, full SHA-256, and `prev` = hash of the previous line (a hash chain). Append-only. |
| `scores.jsonl` | Post-race scores (append-only; the latest line per snapshot wins, in case a result is corrected). |

**Check it yourself:** `python race_weekend.py verify` re-hashes every file and walks the
chain. Or by hand: `shasum -a 256 snapshots/2026/R21/*.json` and compare with `index.jsonl`.
Because this folder is committed to git, the git history is a second, independent record
of when each snapshot appeared.

Stages: `pre_quali` (the pre-lock forecast used for F1 Fantasy), `post_quali` (after
qualifying; `grid_status` says whether the official grid was in), `post_sprint`.
`published_before_lock` is true only if the snapshot was written at least one hour before
the team lock (qualifying start; sprint qualifying on sprint weekends).
