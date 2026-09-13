# ML pipeline

Technical setup for the Pitstop Oracle model and Streamlit dashboard.

**Start here:** [Project README](../README.md) · [Full documentation](../docs/README.md) · [Architecture](../docs/ARCHITECTURE.md)

## Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

| Command | Purpose |
|---|---|
| `python run_pipeline.py` | Live-fetch current season + this weekend (FP1–3, quali, grid), rebuild features, retrain |
| `python run_pipeline.py --all-data` | Clear Jolpica cache for 2022–present and full re-fetch + retrain |
| `python run_pipeline.py --refresh` | Alias for `--all-data` |
| `python run_pipeline.py --skip-ingest` | Retrain from cached raw data |
| `python -c "from f1ml.starting_grid import sync_starting_grid; sync_starting_grid(2026, 14)"` | Refresh starting grid for one GP |
| `streamlit run app.py` | Launch dashboard |
| `python predict_race.py` | CLI weekend card for the next GP on the calendar |
| `python scripts/generate_docs_assets.py` | Regenerate README screenshot PNGs |

## Models (v3)

Two weekend modes, trained separately with a **pairwise ranker + Plackett-Luce** distribution:

| Mode | When | Features |
|---|---|---|
| `pre_quali` | Before qualifying | Car/driver ratings, form, circuit, weather, FP pace |
| `post_quali` | After qualifying | All of the above + quali gap (seconds), grid, sprint |

Win, podium, and expected finish come from a **single** finishing distribution. Constructor pace **resets at 2026**; driver skill carries over. DNF hazard is modeled separately.

Run full ingest (including OpenF1 FP backfill + pit stops):

```bash
python run_pipeline.py              # weekend-first ingest + train (live current season)
python run_pipeline.py --all-data   # full rebuild from 2022
```

After ingest, the pipeline prints a **weekend summary** line (`quali=… fp1/fp2/fp3=… grid=…`) so you can confirm Saturday data landed before opening the dashboard.

## Dashboard pages

| Page | File |
|---|---|
| This Weekend | `pages/1_This_Weekend.py` |
| Race Explorer | `pages/2_Race_Explorer.py` |
| Fantasy Lab | `pages/3_Fantasy_Lab.py` |
| Model Performance | `pages/4_Model_Performance.py` |

## Outputs

| Path | Contents |
|---|---|
| `data/raw/*.parquet` | Races, results, qualifying, standings, weather |
| `data/canonical/*.parquet` | Starting grid, FP pace, weather forecast |
| `data/manual/*.csv` | Grid penalties and fantasy prices overrides |
| `data/processed/driver_race.parquet` | Feature table (one row per driver per race) |
| `models/pre_quali/` | Pre-quali model bundle |
| `models/post_quali/` | Post-quali model bundle |
| `reports/EVAL.md` | Evaluation vs baselines (log-loss, Brier, hit rate) |
| `reports/metrics.json` | Full metrics for Model Performance page |

## Deploy (optional)

[Streamlit Community Cloud](https://streamlit.io/cloud): main file `ml/app.py`, Python 3.11+.
