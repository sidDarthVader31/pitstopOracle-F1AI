# pitstopOracle-F1AI

Predict Formula 1 race winners from historical results (not telemetry).

The first working model lives in [`ml/`](ml/): ingest free Jolpica data, build a driver-race table, beat (or honestly fail to beat) a pole-sitter baseline.

```bash
cd ml && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

See [`ml/README.md`](ml/README.md) and [`ml/reports/EVAL.md`](ml/reports/EVAL.md) after a run.

The TypeScript [`dataCollectionService`](dataCollectionService) is a future Postgres warehouse, not required for v1 training.
