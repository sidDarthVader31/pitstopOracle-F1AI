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
| `python run_pipeline.py` | Ingest Jolpica data + train models |
| `python run_pipeline.py --refresh` | Clear Jolpica cache for ingest years and re-fetch |
| `python run_pipeline.py --skip-ingest` | Retrain from cached raw data |
| `python -c "from f1ml.starting_grid import sync_starting_grid; sync_starting_grid(2026, 13)"` | Refresh starting grid for one GP |
| `streamlit run app.py` | Launch dashboard |
| `python predict_race.py --season 2026 --round 13` | CLI weekend card for one GP |
| `python scripts/generate_docs_assets.py` | Regenerate README screenshot PNGs |

## Models (v2)

Two weekend modes, trained separately:

| Mode | When | Features |
|---|---|---|
| `pre_quali` | Before qualifying | Form, standings, circuit history, weather |
| `post_quali` | After qualifying | All of the above + quali, grid, sprint |

Each mode trains gradient boosting heads for **win**, **podium**, **DNF**, and **expected finish**, with race-level softmax calibration for win probabilities.

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
