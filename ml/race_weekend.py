#!/usr/bin/env python3
"""Race-weekend loop: retrain on every completed race, forecast, then score after the race.

    python race_weekend.py before --season 2026 --round 20   # before lights out
    python race_weekend.py after  --season 2026 --round 20   # once results are published
    python race_weekend.py train  --season 2026 --round 20   # retrain/save models only, no new forecast

`before` uses the same feature recipe (specs.py) every weekend and just adds the
newest race to the training data. Changing the recipe itself is an experiment
that needs walk-forward evidence first (see docs/experiments/).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from f1ml.eval import grid_probabilities, race_log_loss
from f1ml.features import build_driver_race
from f1ml.lifecycle import persist_forecast_snapshot
from f1ml.paths import MODELS, REPORTS
from f1ml.predict import clear_prediction_caches
from f1ml.ranking import WeekendRanker
from f1ml.calibration import ROLLING_MODES, history_rows
from f1ml.train import MODE_RULES, OOS_HISTORY_PATH, _load_training_frame, train_stage_bundle

FORECASTS = REPORTS / "forecasts"
TRACKING = REPORTS / "live_tracking.csv"
PRODUCTION_MC_SAMPLES = 6000

def _stage_for(race: pd.DataFrame) -> tuple[str, str]:
    """Return (model mode, snapshot stage) for the information currently available.

    ``post_sprint`` was trained and evaluated only on races with both a sprint result
    and Grand Prix qualifying. On the 2024+ sprint format the sprint runs *before* GP
    qualifying, so between the two a sprint result alone must not select it: that
    model would have its qualifying and grid inputs imputed. Use ``pre_quali`` (which
    ignores the sprint) until qualifying is known.
    """
    has_quali = race["quali_position"].notna().any()
    has_sprint = pd.to_numeric(race["sprint_position"], errors="coerce").gt(0).any()
    if has_sprint and has_quali:
        return "post_sprint", "post_sprint"
    if has_quali:
        return "post_quali", "post_quali_pre_sprint"
    return "pre_quali", "pre_quali"


def load_oos_history(season: int, round_num: int, mode: str) -> pd.DataFrame | None:
    """Earlier out-of-sample forecasts for one stage, strictly before (season, round)."""
    if not OOS_HISTORY_PATH.exists():
        return None
    h = pd.read_parquet(OOS_HISTORY_PATH)
    h = h[(h["stage"] == mode) & ((h["season"] * 100 + h["round"]) < season * 100 + round_num)]
    return h if len(h) else None


def train_weekend_bundle(
    history: pd.DataFrame, mode: str, season: int, round_num: int | None = None
) -> tuple[WeekendRanker | None, dict]:
    """Live bundle: the same ``train.train_stage_bundle`` recipe the walk-forward evaluates."""
    oos = load_oos_history(season, round_num, mode) if round_num is not None and mode in ROLLING_MODES else None
    if mode in ROLLING_MODES and oos is None:
        print(f"[train] {mode}: no out-of-sample history in {OOS_HISTORY_PATH.name}; "
              "falling back to the last-10-races temperature. Rebuild it with `python run_pipeline.py --skip-ingest`.")
    return train_stage_bundle(history, mode, season, oos_history=oos, mc_samples=PRODUCTION_MC_SAMPLES)


def append_oos_history(race: pd.DataFrame, forecast: pd.DataFrame, stage: str) -> int:
    """Add a scored live forecast (made before the race) to the calibration history."""
    mode = {"pre_quali": "pre_quali", "post_quali_pre_sprint": "post_quali"}.get(stage)
    if mode is None or not {"pace_score", "dnf_prob"} <= set(forecast.columns):
        return 0
    rows = race[["season", "round", "driver_id", "won"]].merge(
        forecast[["driver_id", "pace_score", "dnf_prob"]], on="driver_id", how="inner"
    )
    if rows.empty or rows["won"].sum() != 1:
        return 0
    rows = history_rows(rows, mode)
    if OOS_HISTORY_PATH.exists():
        prev = pd.read_parquet(OOS_HISTORY_PATH)
        same = (prev["stage"] == mode) & (prev["season"] == rows["season"].iloc[0]) & (prev["round"] == rows["round"].iloc[0])
        rows = pd.concat([prev[~same], rows], ignore_index=True)
    rows.sort_values(["stage", "season", "round", "driver_id"]).to_parquet(OOS_HISTORY_PATH, index=False)
    return int(len(rows))


def _archive_current_models(stamp: str) -> Path | None:
    existing = [m for m in MODE_RULES if (MODELS / m).exists()]
    if not existing:
        return None
    archive = MODELS / "archive" / f"auto-{stamp}"
    archive.mkdir(parents=True, exist_ok=True)
    for mode in existing:
        shutil.copytree(MODELS / mode, archive / mode, dirs_exist_ok=True)
    return archive


def _print_card(scored: pd.DataFrame, title: str) -> None:
    view = scored.sort_values("win_prob", ascending=False)
    print(f"\n{title}")
    print(f"{'#':>2}  {'Driver':<18} {'Team':<14} {'Grid':>4} {'Win':>6} {'Podium':>7} {'Points':>7} {'DNF':>5} {'ExpFin':>6}")
    for i, r in enumerate(view.itertuples(index=False), 1):
        name = f"{getattr(r, 'given_name', '')} {getattr(r, 'family_name', '')}".strip() or r.driver_id
        grid = "" if pd.isna(getattr(r, "grid", np.nan)) else int(r.grid)
        print(
            f"{i:>2}  {name[:18]:<18} {str(r.constructor_id)[:14]:<14} {grid!s:>4} "
            f"{r.win_prob:6.1%} {r.podium_prob:7.1%} {r.points_prob:7.1%} {r.dnf_prob:5.0%} {r.expected_finish:6.1f}"
        )


def train_and_save(season: int, round_num: int) -> tuple[dict, dict]:
    """Train every mode on races before (season, round_num) and save them as the live models."""
    history = _load_training_frame()
    history = history[(history["season"] < season) | ((history["season"] == season) & (history["round"] < round_num))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundles, manifest = {}, {"season": season, "round": round_num, "created_at_utc": stamp, "modes": {}}
    for mode in MODE_RULES:
        bundle, info = train_weekend_bundle(history, mode, season, round_num)
        manifest["modes"][mode] = info
        if bundle is not None:
            bundles[mode] = bundle
        print(f"[train] {mode}: {info.get('status')} races={info.get('races', '-')} T={info.get('temperature', '-')} "
              f"({info.get('temperature_source', '-')}) pace_train={info.get('pace_train', '-')}")
    archive = _archive_current_models(stamp)
    for mode, bundle in bundles.items():
        bundle.save(MODELS / mode)
    manifest["previous_models_archived_to"] = str(archive) if archive else None
    (MODELS / "weekend_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Models saved to {MODELS} (previous copy: {archive})")
    return bundles, manifest


def cmd_train(season: int, round_num: int, skip_ingest: bool) -> None:
    if not skip_ingest:
        from f1ml.ingest import run as ingest_run

        ingest_run()
    clear_prediction_caches()
    build_driver_race()
    train_and_save(season, round_num)


def cmd_before(season: int, round_num: int, skip_ingest: bool) -> None:
    if not skip_ingest:
        from f1ml.ingest import run as ingest_run

        ingest_run()
    clear_prediction_caches()
    table = build_driver_race()
    race = table[(table["season"] == season) & (table["round"] == round_num)].copy()
    if race.empty:
        raise SystemExit(f"{season} R{round_num} is not the next race in the data; nothing to forecast.")
    if race["won"].max() == 1:
        raise SystemExit(f"{season} R{round_num} already has a result; use `after` instead.")

    bundles, manifest = train_and_save(season, round_num)
    stamp = manifest["created_at_utc"]
    archive = manifest["previous_models_archived_to"]

    mode, stage = _stage_for(race)
    if mode not in bundles:
        mode, stage = "post_quali", "post_quali_pre_sprint"
    scored = bundles[mode].predict_frame(race)
    title = f"{race['race_name'].iloc[0]} {season} R{round_num} — {mode} forecast (trained through {manifest['modes'][mode]['trained_through']})"
    _print_card(scored, title)
    wet = race["is_wet"].max() == 1 if "is_wet" in race else False
    if wet:
        print("\nNote: rain is forecast. The model has no wet-race training signal; expect more upsets than the probabilities show.")

    snapshot = persist_forecast_snapshot(race, scored, stage, model_version=f"weekend-{stamp}")
    FORECASTS.mkdir(parents=True, exist_ok=True)
    out_csv = FORECASTS / f"{season}_R{round_num:02d}_{stage}.csv"
    keep = ["season", "round", "driver_id", "given_name", "family_name", "constructor_id", "quali_position",
            "grid", "pace_score", "win_prob", "podium_prob", "points_prob", "dnf_prob", "expected_finish"]
    scored[[c for c in keep if c in scored]].sort_values("win_prob", ascending=False).to_csv(out_csv, index=False)
    print(f"\nSaved forecast: {out_csv}\nSnapshot: {snapshot}\nModels saved to {MODELS} (previous copy: {archive})")


def cmd_after(season: int, round_num: int, skip_ingest: bool) -> None:
    if not skip_ingest:
        from f1ml.ingest import run as ingest_run

        ingest_run()
    table = build_driver_race()
    race = table[(table["season"] == season) & (table["round"] == round_num)].copy()
    if race.empty or race["won"].max() != 1:
        raise SystemExit("Results are not published yet — try again later.")
    forecasts = sorted(FORECASTS.glob(f"{season}_R{round_num:02d}_*.csv"))
    if not forecasts:
        raise SystemExit(f"No saved forecast for {season} R{round_num}; run `before` first next time.")
    rows = []
    for path in forecasts:
        fc = pd.read_csv(path)
        added = append_oos_history(race, fc, path.stem.split("_", 2)[-1])
        if added:
            print(f"Calibration history updated from {path.name} ({added} rows)")
        merged = race.merge(fc[["driver_id", "win_prob", "podium_prob", "expected_finish"]], on="driver_id", how="left")
        merged["win_prob"] = merged["win_prob"].fillna(0.0)
        merged = merged.set_index(pd.RangeIndex(len(merged)))
        winner = merged.loc[merged["won"] == 1].iloc[0]
        pick = merged.loc[merged["win_prob"].idxmax()]
        top3 = merged.nlargest(3, "win_prob")["driver_id"].tolist()
        podium_actual = set(merged.loc[merged["position"].between(1, 3), "driver_id"])
        podium_pred = set(merged.nlargest(3, "podium_prob")["driver_id"])
        finished = merged.dropna(subset=["position", "expected_finish"])
        row = {
            "season": season,
            "round": round_num,
            "race": race["race_name"].iloc[0],
            "stage": path.stem.split("_", 2)[-1],
            "actual_winner": winner["driver_id"],
            "model_pick": pick["driver_id"],
            "pick_correct": int(pick["driver_id"] == winner["driver_id"]),
            "winner_in_top3": int(winner["driver_id"] in top3),
            "winner_prob": round(float(winner["win_prob"]), 4),
            "log_loss": round(race_log_loss(merged, merged["win_prob"]), 4),
            "grid_baseline_log_loss": round(race_log_loss(merged, grid_probabilities(merged)), 4),
            "podium_overlap": len(podium_actual & podium_pred),
            "finish_mae": round(float((finished["expected_finish"] - finished["position"]).abs().mean()), 3),
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
    out = pd.DataFrame(rows)
    if TRACKING.exists():
        prev = pd.read_csv(TRACKING)
        prev = prev[~((prev["season"] == season) & (prev["round"] == round_num))]
        out = pd.concat([prev, out], ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(TRACKING, index=False)
    print(f"\nAppended to {TRACKING} ({len(out)} forecasts tracked).")
    print("Next race: run `python race_weekend.py before --season <year> --round <n>` after qualifying.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["before", "after", "train"])
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--skip-ingest", action="store_true", help="Use already-downloaded data")
    a = p.parse_args()
    {"before": cmd_before, "after": cmd_after, "train": cmd_train}[a.command](a.season, a.round, a.skip_ingest)


if __name__ == "__main__":
    main()
