#!/usr/bin/env python3
"""CLI: predict a specific upcoming race weekend."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from f1ml.predict import build_upcoming_race_features, display_columns, get_race_features, score_race


def main() -> None:
    p = argparse.ArgumentParser(description="Predict F1 race winner")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--round", type=int, default=13, help="2026 R13 = Italian GP (Monza)")
    p.add_argument("--model", choices=["rf", "logreg"], default="rf")
    p.add_argument("--wet", action="store_true")
    args = p.parse_args()

    try:
        race_df = get_race_features(args.season, args.round)
        mode = "historical (quali + results available)"
    except ValueError:
        race_df = build_upcoming_race_features(args.season, args.round)
        mode = "pre-weekend (no qualifying yet — form + standings only)"

    if args.wet:
        race_df = race_df.copy()
        race_df["is_wet"] = 1
        race_df["precipitation_mm"] = 5.0

    scored = score_race(race_df, kind=args.model)
    meta = race_df.iloc[0]

    print(f"\n{meta['race_name']} — {meta['circuit_name']} ({meta['date']})")
    print(f"Mode: {mode}")
    print(f"Model: {args.model.upper()}\n")

    top = scored.head(10)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        champ = int(row["champ_position_before"]) if pd.notna(row["champ_position_before"]) else "-"
        print(
            f"{rank:2d}. {row['given_name']} {row['family_name']:<14} "
            f"{row['constructor_name']:<20} {row['win_prob']*100:5.1f}%  (champ P{champ})"
        )

    pick = scored.iloc[0]
    print(f"\nPredicted winner: {pick['given_name']} {pick['family_name']} ({pick['win_prob']*100:.1f}%)")
    if mode.startswith("pre-weekend"):
        print("\nNote: Re-run after Saturday qualifying for a stronger prediction.")


if __name__ == "__main__":
    main()
