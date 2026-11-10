"""Build ledger payloads that follow the forecast output contract.

ML_MODEL_DESIGN_v3 §7 and MVP_SPEC §9.2: event, stage, generated_at, data_as_of,
model_version, feature_snapshot_id, per-driver probabilities, the finishing-position
distribution and projected fantasy points, plus warnings and versions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from f1ml.ledger import LEGACY_STAGE_NAMES, feature_snapshot_id, utc_now
from f1ml.schedule import PRE_LOCK_MARGIN, Weekend
from f1ml.specs import feature_columns

# Until gate G3 passes, the UI shows chance bands, not percentages (MVP_SPEC §5.2).
CALIBRATION_STATUS = "bands_only"

DRIVER_FIELDS = [
    "driver_id", "driver_code", "given_name", "family_name", "constructor_id", "constructor_name",
    "quali_position", "grid", "pace_score", "win_prob", "podium_prob", "points_prob", "dnf_prob",
    "expected_finish",
]


def ledger_stage(stage: str) -> str:
    return LEGACY_STAGE_NAMES.get(stage, stage)


def _finish_dist_columns(scored: pd.DataFrame) -> list[str]:
    cols = [c for c in scored.columns if c.startswith("finish_p") and c.endswith("_prob")]
    return sorted(cols, key=lambda c: int(c[len("finish_p"):-len("_prob")]))


def _feature_records(race: pd.DataFrame, mode: str) -> list[dict]:
    numeric, categorical = feature_columns(mode)
    cols = ["season", "round", "driver_id"] + [c for c in numeric + categorical if c in race.columns]
    return race[list(dict.fromkeys(cols))].to_dict("records")


def compute_data_as_of(table: pd.DataFrame, race: pd.DataFrame, weekend: Weekend | None) -> str:
    """Latest real-world event the inputs include: last finished race, qualifying, official grid."""
    stamps: list[datetime] = []
    done = table[table.groupby(["season", "round"])["won"].transform("max") == 1]
    if not done.empty and "date" in done:
        from f1ml.schedule import get_weekend

        last = done.sort_values(["season", "round"]).iloc[-1]
        wk = get_weekend(int(last["season"]), int(last["round"]))
        stamps.append(wk.race_start if wk and wk.race_start else pd.Timestamp(done["date"].max()).tz_localize("UTC").to_pydatetime())
    if weekend is not None and race["quali_position"].notna().any() and weekend.quali_start:
        stamps.append(weekend.quali_start)
    if "grid_as_of" in race:
        for v in race["grid_as_of"].dropna().astype(str):
            try:
                ts = pd.Timestamp(v)
                stamps.append((ts if ts.tzinfo else ts.tz_localize("UTC")).to_pydatetime())
            except ValueError:
                continue
    latest = max(stamps) if stamps else utc_now()
    return latest.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def build_snapshot_payload(
    *,
    race: pd.DataFrame,
    scored: pd.DataFrame,
    stage: str,
    mode: str,
    model_version: str,
    model_info: dict,
    data_as_of: str,
    weekend: Weekend | None,
    generated_at: datetime | None = None,
    fantasy: Callable[[pd.DataFrame], dict] | None = None,
    warnings: list[str] | None = None,
    grid_meta: dict | None = None,
) -> dict:
    """One frozen forecast. ``fantasy(scored)`` returns {"drivers": {id: {...}}, "constructors": [...], ...}."""
    generated = (generated_at or utc_now()).astimezone(timezone.utc).replace(microsecond=0)
    lock_at = weekend.lock_at if weekend else None
    warnings = list(warnings or [])
    if lock_at is None:
        warnings.append("Team lock time unknown (no schedule row).")

    dist_cols = _finish_dist_columns(scored)
    fantasy_out = fantasy(scored) if fantasy else {}
    fantasy_drivers = fantasy_out.get("drivers", {})

    drivers = []
    scored = scored.reset_index(drop=True)
    scored["sim_index"] = range(len(scored))  # order of the Monte Carlo columns (needed to reproduce draws)
    for _, row in scored.sort_values("win_prob", ascending=False).iterrows():
        rec = {k: row.get(k) for k in DRIVER_FIELDS if k in scored.columns}
        rec["sim_index"] = int(row["sim_index"])
        for k in ("quali_position", "grid"):
            if k in rec and pd.notna(rec[k]):
                rec[k] = int(rec[k])
        rec["finish_position_probs"] = [row[c] for c in dist_cols]  # P(pos=k | classified), k=1..n
        if row["driver_id"] in fantasy_drivers:
            rec["fantasy"] = fantasy_drivers[row["driver_id"]]
        drivers.append(rec)

    grid_known = "grid" in race and race["grid"].notna().any() and race["grid"].gt(0).any()
    grid_official = bool((grid_meta or {}).get("has_grid"))
    event = {
        "season": int(race["season"].iloc[0]),
        "round": int(race["round"].iloc[0]),
        "race_name": str(race["race_name"].iloc[0]),
        "circuit_name": str(race["circuit_name"].iloc[0]) if "circuit_name" in race else None,
        "format": weekend.format if weekend else None,
        "lock_at": lock_at.isoformat() if lock_at else None,
        "quali_start": weekend.quali_start.isoformat() if weekend and weekend.quali_start else None,
        "race_start": weekend.race_start.isoformat() if weekend and weekend.race_start else None,
    }
    payload = {
        "event": event,
        "stage": ledger_stage(stage),
        "model_mode": mode,
        "generated_at": generated.isoformat(),
        "data_as_of": data_as_of,
        "published_before_lock": bool(lock_at and generated <= lock_at - PRE_LOCK_MARGIN) if lock_at else None,
        "grid_status": "official" if grid_official else ("quali_order" if grid_known else "unknown"),
        "grid_as_of": (grid_meta or {}).get("as_of") if grid_official else None,
        "model_version": model_version,
        "model": {k: model_info.get(k) for k in (
            "races", "trained_through", "temperature", "temperature_source", "oos_history_races", "pace_train"
        )},
        "mc_samples": model_info.get("mc_samples"),
        "feature_snapshot_id": feature_snapshot_id(_feature_records(race, mode)),
        "calibration_status": CALIBRATION_STATUS,
        "scoring_rules_version": fantasy_out.get("rules_version"),
        "prices_version": fantasy_out.get("prices_version"),
        "warnings": warnings + list(fantasy_out.get("warnings", [])),
        "drivers": drivers,
        "constructors": fantasy_out.get("constructors", []),
    }
    if fantasy_out.get("oracle_team"):
        payload["oracle_team"] = fantasy_out["oracle_team"]
    return payload
