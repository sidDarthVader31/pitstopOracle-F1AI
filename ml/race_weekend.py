#!/usr/bin/env python3
"""Race-weekend loop: retrain, freeze forecasts in the public ledger, score after the race.

    python race_weekend.py before --season 2026 --round 21   # Thu/Fri, >= 1 h before the team lock
    python race_weekend.py update --season 2026 --round 21   # after qualifying / once the grid is official
    python race_weekend.py after  --season 2026 --round 21   # once results are published
    python race_weekend.py verify                            # check the ledger has not been edited
    python race_weekend.py train  --season 2026 --round 21   # retrain/save models only, no forecast

`before` retrains (approved recipe, weekly refit checks) and freezes a snapshot for the
current stage. `update` freezes a new snapshot with the models `before` saved (no retrain).
Every snapshot is a new, content-hashed file in `ml/ledger/`; nothing is ever overwritten.
Add `--dry-run` to write to a temporary ledger and leave the live models untouched.

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
from f1ml import ledger
from f1ml.forecast_snapshot import build_snapshot_payload, compute_data_as_of, ledger_stage
from f1ml.lifecycle import weekly_refit_checks
from f1ml.paths import MODELS, REPORTS
from f1ml.schedule import get_weekend
from f1ml.starting_grid import grid_metadata
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


def train_and_save(season: int, round_num: int, *, dry_run: bool = False, race_date: str | None = None) -> tuple[dict, dict]:
    """Train every mode on races before (season, round_num).

    The bundles replace the live models only if ``weekly_refit_checks`` passes (and not in a
    dry run). Otherwise they are saved under ``models/candidates/`` and the live models stay.
    """
    history = _load_training_frame()
    history = history[(history["season"] < season) | ((history["season"] == season) & (history["round"] < round_num))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundles, manifest = {}, {"season": season, "round": round_num, "created_at_utc": stamp, "modes": {}}
    for mode in MODE_RULES:
        bundle, info = train_weekend_bundle(history, mode, season, round_num)
        info["mc_samples"] = PRODUCTION_MC_SAMPLES
        manifest["modes"][mode] = info
        if bundle is not None:
            bundles[mode] = bundle
        print(f"[train] {mode}: {info.get('status')} races={info.get('races', '-')} T={info.get('temperature', '-')} "
              f"({info.get('temperature_source', '-')}) pace_train={info.get('pace_train', '-')}")
    problems = weekly_refit_checks(manifest, race_date)
    manifest["refit_checks"] = problems or "passed"
    if dry_run:
        manifest["promoted"] = False
        print(f"[dry-run] refit checks: {problems or 'passed'}; live models in {MODELS} left untouched.")
        return bundles, manifest
    if problems:
        target = MODELS / "candidates" / f"weekend-{stamp}"
        for mode, bundle in bundles.items():
            bundle.save(target / mode)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        manifest["promoted"] = False
        print("[train] REFIT CHECKS FAILED, live models NOT replaced:\n  - " + "\n  - ".join(problems))
        print(f"[train] retrained bundles saved for review in {target}")
        return bundles, manifest
    archive = _archive_current_models(stamp)
    for mode, bundle in bundles.items():
        bundle.save(MODELS / mode)
    manifest["promoted"] = True
    manifest["previous_models_archived_to"] = str(archive) if archive else None
    (MODELS / "weekend_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Models saved to {MODELS} (previous copy: {archive})")
    return bundles, manifest


def _load_race(table: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    race = table[(table["season"] == season) & (table["round"] == round_num)].copy()
    if race.empty:
        raise SystemExit(f"{season} R{round_num} is not the next race in the data; nothing to forecast.")
    if race["won"].max() == 1:
        raise SystemExit(f"{season} R{round_num} already has a result; use `after` instead.")
    return race.reset_index(drop=True)


def _ingest(skip_ingest: bool) -> None:
    if not skip_ingest:
        from f1ml.ingest import run as ingest_run

        ingest_run()
    clear_prediction_caches()


def _fantasy_hook(season: int, round_num: int, bundle: WeekendRanker, race: pd.DataFrame):
    from f1ml.fantasy import snapshot_fantasy

    return lambda scored: snapshot_fantasy(scored, season=season, round_num=round_num, bundle=bundle, race=race)


def snapshot_forecast(
    *,
    table: pd.DataFrame,
    race: pd.DataFrame,
    season: int,
    round_num: int,
    bundles: dict,
    model_infos: dict,
    model_version: str,
    ledger_root: Path,
) -> ledger.SnapshotRef:
    """Score the race with the bundle for the current stage and freeze it in the ledger."""
    mode, stage = _stage_for(race)
    if mode not in bundles:
        mode, stage = "post_quali", "post_quali_pre_sprint"
    bundle = bundles[mode]
    scored = bundle.predict_frame(race)
    info = dict(model_infos.get(mode) or {})
    info.setdefault("mc_samples", bundle.mc_samples)
    info.setdefault("temperature", bundle.temperature)
    title = f"{race['race_name'].iloc[0]} {season} R{round_num} — {mode} forecast (trained through {info.get('trained_through', '?')})"
    _print_card(scored, title)
    warnings = []
    if "is_wet" in race and race["is_wet"].max() == 1:
        warnings.append("Rain is forecast. The model has no wet-race training signal; expect more upsets than the chances show.")
        print("\nNote: " + warnings[-1])
    weekend = get_weekend(season, round_num)
    payload = build_snapshot_payload(
        race=race,
        scored=scored,
        stage=stage,
        mode=mode,
        model_version=model_version,
        model_info=info,
        data_as_of=compute_data_as_of(table, race, weekend),
        weekend=weekend,
        fantasy=_fantasy_hook(season, round_num, bundle, race),
        warnings=warnings,
        grid_meta=grid_metadata(season, round_num),
    )
    ref = ledger.write_snapshot(payload, ledger_root)
    lock = payload["event"].get("lock_at")
    timing = {True: "published >= 1 h before the team lock", False: "published AFTER the pre-lock deadline", None: "lock time unknown"}
    print(f"\nFrozen snapshot {ref.snapshot_id} ({timing[payload['published_before_lock']]}; lock {lock})")
    print(f"  file:   {ref.path}\n  sha256: {ref.sha256}")
    return ref


def cmd_train(season: int, round_num: int, skip_ingest: bool, **_) -> None:
    _ingest(skip_ingest)
    build_driver_race()
    train_and_save(season, round_num)


def cmd_before(season: int, round_num: int, skip_ingest: bool, dry_run: bool = False, ledger_root: Path | None = None) -> None:
    _ingest(skip_ingest)
    table = build_driver_race()
    race = _load_race(table, season, round_num)
    ledger_root = _ledger_root(ledger_root, dry_run)
    _require_intact(ledger_root)
    weekend = get_weekend(season, round_num)
    race_date = weekend.race_start.date().isoformat() if weekend and weekend.race_start else str(race["date"].iloc[0])[:10]
    bundles, manifest = train_and_save(season, round_num, dry_run=dry_run, race_date=race_date)
    if manifest.get("promoted") or dry_run:
        version = f"weekend-{manifest['created_at_utc']}" + ("" if manifest.get("promoted") else "-dryrun")
        infos = manifest["modes"]
    else:
        print("[before] forecasting with the existing live models instead.")
        bundles, infos, version = _load_live_bundles()
    snapshot_forecast(table=table, race=race, season=season, round_num=round_num, bundles=bundles,
                      model_infos=infos, model_version=version, ledger_root=ledger_root)


def cmd_update(season: int, round_num: int, skip_ingest: bool, dry_run: bool = False, ledger_root: Path | None = None) -> None:
    """New snapshot for the current stage (after qualifying, after the official grid) without retraining."""
    _ingest(skip_ingest)
    table = build_driver_race()
    race = _load_race(table, season, round_num)
    ledger_root = _ledger_root(ledger_root, dry_run)
    _require_intact(ledger_root)
    bundles, infos, version = _load_live_bundles()
    trained_for = (infos.get("_manifest") or {})
    if (trained_for.get("season"), trained_for.get("round")) != (season, round_num):
        print(f"[update] WARNING: live models were trained for {trained_for.get('season')} R{trained_for.get('round')}, "
              f"not {season} R{round_num}. Run `before` first unless this is intended.")
    snapshot_forecast(table=table, race=race, season=season, round_num=round_num, bundles=bundles,
                      model_infos=infos, model_version=version, ledger_root=ledger_root)


def _load_live_bundles() -> tuple[dict, dict, str]:
    manifest_path = MODELS / "weekend_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    bundles = {m: WeekendRanker.load(MODELS / m) for m in MODE_RULES if (MODELS / m / "meta.joblib").exists()}
    infos = dict(manifest.get("modes", {}))
    infos["_manifest"] = {"season": manifest.get("season"), "round": manifest.get("round")}
    version = f"weekend-{manifest.get('created_at_utc', 'unknown')}"
    return bundles, infos, version


def _ledger_root(ledger_root: Path | None, dry_run: bool) -> Path:
    if ledger_root is not None:
        return ledger_root
    if dry_run:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="ledger-dryrun-"))
        print(f"[dry-run] writing snapshots to a temporary ledger: {root}")
        return root
    return ledger.LEDGER_ROOT


def _require_intact(root: Path) -> None:
    problems = ledger.verify_ledger(root)
    if problems:
        raise SystemExit("Ledger verification failed; fix before writing:\n  " + "\n  ".join(problems))


def _forecasts_to_score(season: int, round_num: int, ledger_root: Path) -> list[dict]:
    """Frozen forecasts for a race: ledger snapshots, else legacy reports/forecasts CSVs."""
    out = []
    for entry in ledger.snapshots_for(season, round_num, ledger_root):
        payload = ledger.load_snapshot(entry, ledger_root)
        out.append({"stage": entry["stage"], "frame": pd.DataFrame(payload["drivers"]), "entry": entry, "payload": payload})
    if not out:
        for path in sorted(FORECASTS.glob(f"{season}_R{round_num:02d}_*.csv")):
            out.append({"stage": ledger_stage(path.stem.split("_", 2)[-1]), "frame": pd.read_csv(path),
                        "entry": None, "payload": None, "legacy_file": path.name})
    return out


def score_forecast(race: pd.DataFrame, fc: pd.DataFrame) -> dict:
    """Score one frozen forecast against the official result."""
    merged = race.merge(fc[["driver_id", "win_prob", "podium_prob", "expected_finish"]], on="driver_id", how="left")
    merged["win_prob"] = merged["win_prob"].fillna(0.0)
    merged["podium_prob"] = merged["podium_prob"].fillna(0.0)
    merged = merged.set_index(pd.RangeIndex(len(merged)))
    winner = merged.loc[merged["won"] == 1].iloc[0]
    pick = merged.loc[merged["win_prob"].idxmax()]
    top3 = merged.nlargest(3, "win_prob")["driver_id"].tolist()
    podium_actual = set(merged.loc[merged["position"].between(1, 3), "driver_id"])
    podium_pred = set(merged.nlargest(3, "podium_prob")["driver_id"])
    finished = merged.dropna(subset=["position", "expected_finish"])
    return {
        "actual_winner": winner["driver_id"],
        "model_pick": pick["driver_id"],
        "pick_correct": int(pick["driver_id"] == winner["driver_id"]),
        "winner_in_top3": int(winner["driver_id"] in top3),
        "winner_prob": round(float(winner["win_prob"]), 4),
        "log_loss": round(race_log_loss(merged, merged["win_prob"]), 4),
        "grid_baseline_log_loss": round(race_log_loss(merged, grid_probabilities(merged)), 4),
        "podium_overlap": len(podium_actual & podium_pred),
        "finish_mae": round(float((finished["expected_finish"] - finished["position"]).abs().mean()), 3),
    }


def cmd_after(season: int, round_num: int, skip_ingest: bool, dry_run: bool = False, ledger_root: Path | None = None) -> None:
    _ingest(skip_ingest)
    table = build_driver_race()
    race = table[(table["season"] == season) & (table["round"] == round_num)].copy()
    if race.empty or race["won"].max() != 1:
        raise SystemExit("Results are not published yet — try again later.")
    ledger_root = ledger_root or ledger.LEDGER_ROOT
    _require_intact(ledger_root)
    forecasts = _forecasts_to_score(season, round_num, ledger_root)
    if not forecasts:
        raise SystemExit(f"No saved forecast for {season} R{round_num}; run `before` first next time.")
    weekend = get_weekend(season, round_num)
    race_start = weekend.race_start.isoformat() if weekend and weekend.race_start else None

    # Calibration history: only the latest snapshot per stage made before the race started.
    latest_per_stage: dict[str, dict] = {}
    for f in forecasts:
        made = f["entry"]["generated_at"] if f["entry"] else None
        if race_start is None or made is None or made < race_start:
            latest_per_stage[f["stage"]] = f
    rows = []
    for f in forecasts:
        if not dry_run and latest_per_stage.get(f["stage"]) is f:
            added = append_oos_history(race, f["frame"], "post_quali_pre_sprint" if f["stage"] == "post_quali" else f["stage"])
            if added:
                print(f"Calibration history updated from {f['stage']} forecast ({added} rows)")
        row = {
            "season": season,
            "round": round_num,
            "race": race["race_name"].iloc[0],
            "stage": f["stage"],
            "snapshot_id": f["entry"]["snapshot_id"] if f["entry"] else None,
            "generated_at": f["entry"]["generated_at"] if f["entry"] else None,
            "published_before_lock": f["entry"].get("published_before_lock") if f["entry"] else None,
            **score_forecast(race, f["frame"]),
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if f["payload"] is not None:
            from f1ml.fantasy import score_snapshot_fantasy

            row.update(score_snapshot_fantasy(f["payload"], race, table))
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))
        if f["entry"] and not dry_run:
            ledger.append_score(row, ledger_root)
    if dry_run:
        print("[dry-run] nothing written.")
        return
    out = pd.DataFrame(rows)
    if TRACKING.exists():
        prev = pd.read_csv(TRACKING)
        prev = prev[~((prev["season"] == season) & (prev["round"] == round_num))]
        out = pd.concat([prev, out], ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(TRACKING, index=False)
    print(f"\nScores appended to {ledger_root / 'scores.jsonl'} and {TRACKING} ({len(out)} forecasts tracked).")
    print("Next race: `python race_weekend.py before --season <year> --round <n>` (Thu/Fri, before the lock).")


def cmd_verify(**_) -> None:
    problems = ledger.verify_ledger(ledger.LEDGER_ROOT)
    n = len(ledger.read_index(ledger.LEDGER_ROOT))
    if problems:
        raise SystemExit("Ledger verification FAILED:\n  " + "\n  ".join(problems))
    print(f"Ledger OK: {n} snapshots, hashes and hash chain intact ({ledger.LEDGER_ROOT}).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["before", "update", "after", "train", "verify"])
    p.add_argument("--season", type=int)
    p.add_argument("--round", type=int)
    p.add_argument("--skip-ingest", action="store_true", help="Use already-downloaded data")
    p.add_argument("--dry-run", action="store_true",
                   help="Write snapshots to a temporary ledger; never touch live models, tracking or calibration files")
    a = p.parse_args()
    if a.command == "verify":
        return cmd_verify()
    if a.season is None or a.round is None:
        p.error("--season and --round are required")
    cmds = {"before": cmd_before, "update": cmd_update, "after": cmd_after, "train": cmd_train}
    cmds[a.command](a.season, a.round, a.skip_ingest, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
