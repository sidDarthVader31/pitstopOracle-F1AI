"""Versioned candidate training and immutable forecast snapshots."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd

from f1ml.paths import MODELS, PROCESSED, RAW
from f1ml.ranking import WeekendRanker, predict_dnf_probs, predict_scores, train_dnf_model, train_pairwise_ranker
from f1ml.specs import WeekendMode, feature_columns

REQUIRED_PROMOTION_MODES = ("pre_quali", "post_quali", "post_sprint")


def promotion_eligibility(evidence: dict) -> tuple[bool, list[str]]:
    """Apply the saved chronological-OOS champion promotion criteria."""
    reasons = []
    if evidence.get("evaluation_type") != "chronological_out_of_sample":
        reasons.append("evaluation is not marked chronological out-of-sample")
    if int(evidence.get("race_count", 0)) < int(evidence.get("minimum_races", 30)):
        reasons.append("insufficient held-out race coverage")
    candidate_ll = evidence.get("candidate_win_log_loss")
    champion_ll = evidence.get("champion_win_log_loss")
    if candidate_ll is None or champion_ll is None or candidate_ll >= champion_ll:
        reasons.append("candidate does not beat champion on win log loss")
    candidate_brier = evidence.get("candidate_brier")
    champion_brier = evidence.get("champion_brier")
    if candidate_brier is None or champion_brier is None or candidate_brier > champion_brier + 0.005:
        reasons.append("candidate materially worsens or omits Brier score")
    candidate_cal = evidence.get("candidate_calibration_error")
    champion_cal = evidence.get("champion_calibration_error")
    if candidate_cal is None or champion_cal is None or candidate_cal > champion_cal + 0.01:
        reasons.append("candidate materially worsens or omits calibration error")
    if not evidence.get("regulation_era_results"):
        reasons.append("regulation-era results are missing")
    return not reasons, reasons


def promote_candidate(version: str, evidence_path: Path) -> Path:
    """Promote a fully evidenced candidate, preserving the prior champion for rollback."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ValueError("version must be a 1–64 character filesystem-safe label")
    candidate = MODELS / "candidates" / version
    manifest_path = candidate / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"candidate manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("promoted"):
        raise ValueError("candidate has already been promoted")
    missing = [mode for mode in REQUIRED_PROMOTION_MODES if manifest.get("modes", {}).get(mode, {}).get("status") != "candidate"]
    if missing:
        raise ValueError(f"candidate is missing trained modes: {', '.join(missing)}")
    evidence = json.loads(evidence_path.read_text())
    eligible, reasons = promotion_eligibility(evidence)
    if not eligible:
        raise ValueError("promotion rejected: " + "; ".join(reasons))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_root = MODELS / "promotion_records"
    record_root.mkdir(parents=True, exist_ok=True)
    archive = MODELS / "archive" / f"before-{version}-{timestamp}"
    archive.mkdir(parents=True, exist_ok=False)
    staging = Path(mkdtemp(prefix=f"promotion-{version}-", dir=MODELS))
    replaced: list[str] = []
    try:
        for mode in REQUIRED_PROMOTION_MODES:
            shutil.copytree(candidate / mode, staging / mode)
        for mode in REQUIRED_PROMOTION_MODES:
            champion_dir = MODELS / mode
            if champion_dir.exists():
                shutil.copytree(champion_dir, archive / mode)
        for mode in REQUIRED_PROMOTION_MODES:
            destination = MODELS / mode
            backup = staging / f"{mode}.previous"
            if destination.exists():
                destination.replace(backup)
            (staging / mode).replace(destination)
            replaced.append(mode)
        record = {
            "version": version,
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence": evidence,
            "candidate_manifest": manifest,
            "rollback_archive": str(archive.relative_to(MODELS)),
            "previous_champion_archived": bool(any(archive.iterdir())),
        }
        (record_root / f"{version}-{timestamp}.json").write_text(json.dumps(record, indent=2))
        manifest["promoted"] = True
        manifest["promotion_record"] = str((record_root / f"{version}-{timestamp}.json").relative_to(MODELS))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except Exception:
        for mode in reversed(replaced):
            destination = MODELS / mode
            if destination.exists():
                shutil.rmtree(destination)
            backup = staging / f"{mode}.previous"
            if backup.exists():
                backup.replace(destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return record_root / f"{version}-{timestamp}.json"


def train_candidate(version: str) -> Path:
    """Train versioned, non-promoted bundles using a trailing calibration window."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ValueError("version must be a 1–64 character filesystem-safe label")
    target = MODELS / "candidates" / version
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    df = pd.read_parquet(PROCESSED / "driver_race.parquet")
    df = df[(df.groupby(["season", "round"])["won"].transform("max") == 1) & df["dnf"].notna()].copy()
    manifest = {
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "promoted": False,
        "training_date_range": [str(df["date"].min()), str(df["date"].max())],
        "modes": {},
    }
    for mode in ("pre_quali", "post_quali", "post_sprint"):
        if mode == "pre_quali":
            eligible = df
            calibration_races = 10
        elif mode == "post_quali":
            eligible = df[df["quali_position"].notna()].copy()
            calibration_races = 10
        else:
            eligible = df[pd.to_numeric(df["sprint_position"], errors="coerce") > 0].copy()
            calibration_races = 5
        keys = sorted(eligible.groupby(["season", "round"]).groups)
        min_races = calibration_races + 10
        if len(keys) < min_races:
            manifest["modes"][mode] = {"status": "skipped", "reason": "insufficient eligible races"}
            continue
        val_keys = set(keys[-calibration_races:])
        fit_keys = set(keys[:-calibration_races])
        val = eligible[
            eligible.apply(lambda row: (row["season"], row["round"]) in val_keys, axis=1)
        ]
        fit = eligible[
            eligible.apply(lambda row: (row["season"], row["round"]) in fit_keys, axis=1)
        ]
        ranker = train_pairwise_ranker(fit, mode)
        dnf_model, dnf_features = train_dnf_model(fit, mode)
        scores = predict_scores(ranker, val, mode)
        risks = predict_dnf_probs(dnf_model, val, dnf_features)
        from f1ml.train import _fit_plackett_luce_temperature

        temperature = _fit_plackett_luce_temperature(val, scores, risks)
        bundle = WeekendRanker(mode, ranker, dnf_model, dnf_features, temperature, mc_samples=1200)
        bundle.save(target / mode)
        numeric, categorical = feature_columns(mode)
        manifest["modes"][mode] = {
            "status": "candidate",
            "fit_races": len(fit_keys),
            "calibration_races": len(val_keys),
            "fit_through": str(fit["date"].max()),
            "calibrated_through": str(val["date"].max()),
            "features_sha256": hashlib.sha256(
                json.dumps({"numeric": numeric, "categorical": categorical}, sort_keys=True).encode()
            ).hexdigest(),
        }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return target


def persist_forecast_snapshot(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    forecast_stage: str,
    model_version: str,
    output_root: Path | None = None,
) -> Path:
    """Write an immutable feature/prediction snapshot and a source manifest."""
    if forecast_stage not in {"pre_quali", "post_quali_pre_sprint", "post_sprint"}:
        raise ValueError(f"unsupported forecast stage: {forecast_stage}")
    if features.empty or predictions.empty:
        raise ValueError("features and predictions must both contain the full race field")
    if features[["season", "round"]].drop_duplicates().shape[0] != 1:
        raise ValueError("a snapshot must contain exactly one race")
    key_cols = ["season", "round", "driver_id"]
    if features.duplicated(key_cols).any() or predictions.duplicated(key_cols).any():
        raise ValueError("snapshot inputs must have one row per driver and race")
    prediction_columns = key_cols + [
        c
        for c in predictions.columns
        if c in {
            "pace_score", "dnf_prob", "win_prob", "podium_prob", "points_prob",
            "expected_finish", "finish_position_probs", "win_prob_softmax",
        }
        or (c.startswith("finish_p") and c.endswith("_prob"))
    ]
    prediction_view = predictions[list(dict.fromkeys(prediction_columns))].copy()
    merged = features.merge(
        prediction_view,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(predictions):
        raise ValueError("feature and prediction fields do not match")
    now = datetime.now(timezone.utc)
    snapshot_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    root = output_root or (RAW.parent / "snapshots" / "forecasts")
    season, round_num = int(merged["season"].iloc[0]), int(merged["round"].iloc[0])
    destination = root / str(season) / f"R{round_num:02d}" / forecast_stage / snapshot_id
    destination.mkdir(parents=True, exist_ok=False)
    outcome_columns = [
        "position", "position_text", "status", "won", "podium", "points_finish",
        "dnf", "points", "laps", "time_ms", "fastest_lap_rank",
    ]
    snapshot = merged.drop(columns=outcome_columns, errors="ignore")
    snapshot["forecast_stage"] = forecast_stage
    snapshot["forecast_cutoff_at"] = now.isoformat()
    snapshot["model_version"] = model_version
    snapshot.to_parquet(destination / "forecast.parquet", index=False)
    spec_mode: WeekendMode = (
        "post_quali" if forecast_stage == "post_quali_pre_sprint" else forecast_stage
    )
    manifest = {
        "snapshot_id": snapshot_id,
        "season": season,
        "round": round_num,
        "forecast_stage": forecast_stage,
        "forecast_cutoff_at": now.isoformat(),
        "model_version": model_version,
        "feature_spec_sha256": hashlib.sha256(
            json.dumps(feature_columns(spec_mode), sort_keys=True).encode()
        ).hexdigest(),
        "source_timestamps": {
            "starting_grid_as_of": sorted(merged["grid_as_of"].dropna().astype(str).unique())
            if "grid_as_of" in merged
            else [],
            "weather_forecast_sources": sorted(merged["weather_source"].dropna().astype(str).unique())
            if "weather_source" in merged
            else [],
        },
        "feature_rows": len(snapshot),
        "file": "forecast.parquet",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return destination
