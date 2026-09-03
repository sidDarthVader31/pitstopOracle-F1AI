# F1 winner model (v1)

Beginner pipeline: free Jolpica results + Open-Meteo weather → one row per driver per race → baselines vs sklearn.

Telemetry / FastF1 lap data is intentionally not used here.

## Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py
```

Re-run training only (after ingest cache exists):

```bash
python run_pipeline.py --skip-ingest
```

Writes:

- `data/raw/*.parquet` — races, results, qualifying, sprints, standings, weather
- `data/processed/driver_race.parquet` — feature table
- `models/winner_rf.joblib` — random forest
- `reports/EVAL.md` — pole vs championship vs model hit rates
