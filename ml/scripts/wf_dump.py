#!/usr/bin/env python3
"""Run the walk-forward evaluation once and cache every test race's per-driver outputs.

    python scripts/wf_dump.py baseline                # current recipe
    F1ML_PACE_TRAIN=classified python scripts/wf_dump.py pace_classified

Writes ``data/cache/experiments/<tag>.parquet`` (git-ignored) with pace scores, DNF
probabilities, the fold's temperature and the full finishing-position distribution
for every stage, plus ``<tag>_metrics.json`` with the walk-forward summary. The
experiment scripts (``scripts/exp_*.py``) re-score these files offline, so a
calibration method can be tried without refitting every fold. Same folds and code
as ``reports/EVAL.md``; no model artifact is written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from f1ml.paths import CACHE  # noqa: E402
from f1ml.train import train_and_eval  # noqa: E402

OUT = CACHE / "experiments"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    metrics = train_and_eval(write_reports=False, predictions_out=rows)
    pd.DataFrame(rows).to_parquet(OUT / f"{tag}.parquet", index=False)
    summary = {"tag": tag, "seconds": round(time.time() - t0, 1), "oos_all": metrics.get("oos_all"),
               "oos_2026": metrics.get("oos"), "walk_forward": metrics.get("walk_forward")}
    (OUT / f"{tag}_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("tag", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
