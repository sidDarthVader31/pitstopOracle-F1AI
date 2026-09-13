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
from f1ml.predict import clear_prediction_caches
from f1ml.train import recalibrate_temperatures, train_and_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument(
        "--all-data",
        action="store_true",
        help="Clear Jolpica cache for the ingest year range and re-fetch all data from API",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Alias for --all-data (full re-fetch from first season)",
    )
    p.add_argument(
        "--recalibrate-only",
        action="store_true",
        help="Refit win-model temperature on existing joblibs (no retrain)",
    )
    args = p.parse_args()

    if args.recalibrate_only:
        results = recalibrate_temperatures()
        print("recalibrated temperatures:", results)
        return

    all_data = args.all_data or args.refresh
    if not args.skip_ingest:
        ingest_run(args.start_year, args.end_year, refresh=all_data, all_data=all_data)
    clear_prediction_caches()
    df = build_driver_race()
    print(f"feature table rows={len(df)} races={df.groupby(['season','round']).ngroups}")
    metrics = train_and_eval()
    wf26 = metrics.get("walk_forward_2026", [])
    post_hits = [r["post_hit"] for r in wf26 if "post_hit" in r]
    post_ll = [r["post_log_loss"] for r in wf26 if "post_log_loss" in r]
    qg_ll = [r["post_quali_gap_log_loss"] for r in wf26 if "post_quali_gap_log_loss" in r]
    if wf26:
        print(
            "2026 walk-forward post_quali hit",
            round(sum(post_hits) / len(post_hits), 4) if post_hits else "n/a",
        )
        print(
            "2026 walk-forward post_quali log_loss",
            round(sum(post_ll) / len(post_ll), 4) if post_ll else "n/a",
        )
        print(
            "2026 walk-forward quali-gap log_loss",
            round(sum(qg_ll) / len(qg_ll), 4) if qg_ll else "n/a",
        )
    else:
        train_post = metrics.get("train_post_quali", {})
        print("train post_quali hit", round(train_post.get("train_post_quali_hit", 0), 4))
        print("train post_quali log_loss", round(train_post.get("train_post_quali_log_loss", 0), 4))
    print("wrote ml/reports/EVAL.md")


if __name__ == "__main__":
    main()
