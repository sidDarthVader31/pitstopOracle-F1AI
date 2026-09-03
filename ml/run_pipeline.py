#!/usr/bin/env python3
"""Run ingest → features → train/eval."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from f1ml.features import build_driver_race
from f1ml.ingest import run as ingest_run
from f1ml.paths import END_YEAR, START_YEAR
from f1ml.train import train_and_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument("--skip-ingest", action="store_true")
    args = p.parse_args()

    if not args.skip_ingest:
        ingest_run(args.start_year, args.end_year)
    df = build_driver_race()
    print(f"feature table rows={len(df)} races={df.groupby(['season','round']).ngroups}")
    metrics = train_and_eval()
    print("test pole hit", round(metrics["test_pole_hit"], 4))
    print("test champ hit", round(metrics["test_champ_hit"], 4))
    print("test rf hit", round(metrics["test_rf_hit"], 4))
    print("test logreg hit", round(metrics["test_logreg_hit"], 4))
    print("wrote ml/reports/EVAL.md")


if __name__ == "__main__":
    main()
