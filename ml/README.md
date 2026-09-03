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
| `python run_pipeline.py --skip-ingest` | Retrain from cached raw data |
| `streamlit run app.py` | Launch dashboard |
| `python predict_race.py --season 2026 --round 13` | CLI prediction for one GP |
| `python scripts/generate_docs_assets.py` | Regenerate README screenshot PNGs |

## Dashboard pages

See [docs/README.md](../docs/README.md#dashboard-tour) for screenshots.

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
| `data/processed/driver_race.parquet` | Feature table (one row per driver per race) |
| `models/winner_rf.joblib` | Random forest classifier |
| `reports/EVAL.md` | Evaluation vs pole baseline |

## Deploy (optional)

[Streamlit Community Cloud](https://streamlit.io/cloud): main file `ml/app.py`, Python 3.11+.
