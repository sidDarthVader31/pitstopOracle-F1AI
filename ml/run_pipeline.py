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
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Clear Jolpica disk cache for the ingest year range and re-fetch from API",
    )
    args = p.parse_args()

    if not args.skip_ingest:
        ingest_run(args.start_year, args.end_year, refresh=args.refresh)
    df = build_driver_race()
    print(f"feature table rows={len(df)} races={df.groupby(['season','round']).ngroups}")
    metrics = train_and_eval()
    print("test pre_quali hit", round(metrics.get("test_pre_quali", {}).get("test_pre_quali_hit", 0), 4))
    print("test post_quali hit", round(metrics.get("test_post_quali", {}).get("test_post_quali_hit", 0), 4))
    print("test post_quali log_loss", round(metrics.get("test_post_quali", {}).get("test_post_quali_log_loss", 0), 4))
    print("wrote ml/reports/EVAL.md")


if __name__ == "__main__":
    main()
