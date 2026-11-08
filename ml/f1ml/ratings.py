"""Rolling constructor pace and driver skill ratings with era reset."""

from __future__ import annotations

import numpy as np
import pandas as pd

ERA_RESET_YEAR = 2026
NEW_ENTRANT_CAR_PRIOR = 0.0  # seconds behind pole (neutral)
NEW_ENTRANT_DRIVER_PRIOR = 0.0  # seconds vs teammate
CAR_EWMA_ALPHA = 0.35
DRIVER_EWMA_ALPHA = 0.30
DNF_EWMA_ALPHA = 0.25


def _era(season: int) -> int:
    return 1 if season >= ERA_RESET_YEAR else 0


def _shifted_ewma(
    series: pd.Series,
    alpha: float,
    prior: float,
    reset_mask: pd.Series | None = None,
) -> pd.Series:
    """Exponentially weighted mean using only prior observations."""
    out = np.empty(len(series), dtype=float)
    state = prior
    for i, val in enumerate(series):
        out[i] = state
        if pd.notna(val):
            if reset_mask is not None and bool(reset_mask.iloc[i]):
                state = prior
            state = alpha * float(val) + (1.0 - alpha) * state
    return pd.Series(out, index=series.index)


def _teammate_quali_gap_seconds(df: pd.DataFrame) -> pd.Series:
    """Driver quali gap to teammate in seconds (positive = slower)."""
    gaps = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, _), g in df.groupby(["season", "round"], sort=False):
        sub = g.dropna(subset=["best_quali_seconds"])
        if sub.empty:
            continue
        for cid, team in sub.groupby("constructor_id", sort=False):
            if len(team) < 2:
                continue
            best = team["best_quali_seconds"].min()
            for idx in team.index:
                gaps.loc[idx] = float(df.loc[idx, "best_quali_seconds"]) - float(best)
    return gaps


def _teammate_race_gap(df: pd.DataFrame) -> pd.Series:
    """Race position minus teammate best (positive = worse)."""
    gaps = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, _), g in df.groupby(["season", "round"], sort=False):
        sub = g.dropna(subset=["position"])
        if sub.empty:
            continue
        for _, team in sub.groupby("constructor_id", sort=False):
            if len(team) < 2:
                continue
            best = team["position"].min()
            for idx in team.index:
                gaps.loc[idx] = float(df.loc[idx, "position"]) - float(best)
    return gaps


def add_pace_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Quali / FP / race pace deltas vs field pole or median."""
    out = df.copy()
    out["regs_era"] = out["season"].map(_era)

    # Quali gap to pole (seconds).
    pole_times = (
        out.groupby(["season", "round"])["best_quali_seconds"]
        .transform("min")
    )
    out["quali_gap_s"] = out["best_quali_seconds"] - pole_times
    out["quali_gap_to_median_s"] = out["best_quali_seconds"] - out.groupby(
        ["season", "round"]
    )["best_quali_seconds"].transform("median")

    # FP long-run: use fp_longrun_delta if present else fp2/fp3 best deltas.
    if "fp_longrun_delta" not in out.columns:
        out["fp_longrun_delta"] = out.get("fp3_best_lap_delta", np.nan)
        out["fp_longrun_delta"] = out["fp_longrun_delta"].fillna(
            out.get("fp2_best_lap_delta", np.nan)
        )
    if "fp_best_delta" not in out.columns:
        out["fp_best_delta"] = out.get("fp3_best_lap_delta", np.nan)
        out["fp_best_delta"] = out["fp_best_delta"].fillna(
            out.get("fp2_best_lap_delta", np.nan)
        )

    out["teammate_quali_gap_s"] = _teammate_quali_gap_seconds(out)
    out["teammate_race_gap"] = _teammate_race_gap(out)
    return out


def latest_ratings_before_round(
    history: pd.DataFrame, season: int, round_num: int
) -> pd.DataFrame:
    """Latest car/driver ratings from completed races strictly before round_num."""
    prior = history[(history["season"] == season) & (history["round"] < round_num)].copy()
    if prior.empty:
        prior = history[history["season"] < season].copy()
    if prior.empty:
        return pd.DataFrame()
    rated = build_ratings(prior)
    latest_round = int(rated["round"].max())
    snap = rated[rated["round"] == latest_round][
        [
            "driver_id",
            "constructor_id",
            "car_pace_ewma",
            "driver_skill_ewma",
            "car_driver_combo",
            "dnf_hazard_car",
            "dnf_hazard_driver",
            "dnf_hazard_combined",
        ]
    ].drop_duplicates("driver_id")
    return snap


def build_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add shifted EWMA ratings entering each race (no leakage).
    Constructor pace resets at ERA_RESET_YEAR; driver skill carries across eras.
    """
    out = add_pace_gaps(df)
    out = out.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)

    car_rating = pd.Series(NEW_ENTRANT_CAR_PRIOR, index=out.index, dtype=float)
    driver_rating = pd.Series(NEW_ENTRANT_DRIVER_PRIOR, index=out.index, dtype=float)
    dnf_hazard_car = pd.Series(0.15, index=out.index, dtype=float)
    dnf_hazard_driver = pd.Series(0.10, index=out.index, dtype=float)

    # Per-constructor / per-driver histories within era.
    for constructor_id, cgrp in out.groupby("constructor_id", sort=False):
        cgrp = cgrp.sort_values(["season", "round"])
        car_state = NEW_ENTRANT_CAR_PRIOR
        dnf_car_state = 0.15
        prev_era = None
        for (_, _), event in cgrp.groupby(["season", "round"], sort=True):
            row = event.iloc[0]
            if prev_era is not None and row["season"] >= ERA_RESET_YEAR and prev_era < ERA_RESET_YEAR:
                car_state = NEW_ENTRANT_CAR_PRIOR
                dnf_car_state = 0.15
            car_rating.loc[event.index] = car_state
            dnf_hazard_car.loc[event.index] = dnf_car_state
            prev_era = row["season"]

            pace_cols = ["quali_gap_s", "fp_longrun_delta", "race_pace_delta"]
            pace_parts = [
                float(value)
                for col in pace_cols
                if col in event
                for value in event[col].dropna().tolist()
            ]
            if pace_parts:
                pace_obs = float(np.mean(pace_parts))
                car_state = CAR_EWMA_ALPHA * pace_obs + (1 - CAR_EWMA_ALPHA) * car_state
            event_dnf = event["dnf"].dropna() if "dnf" in event else pd.Series(dtype=float)
            if not event_dnf.empty:
                dnf_car_state = DNF_EWMA_ALPHA * float(event_dnf.mean()) + (1 - DNF_EWMA_ALPHA) * dnf_car_state

    for driver_id, dgrp in out.groupby("driver_id", sort=False):
        dgrp = dgrp.sort_values(["season", "round"])
        drv_state = NEW_ENTRANT_DRIVER_PRIOR
        dnf_drv_state = 0.10
        for idx, row in dgrp.iterrows():
            driver_rating.loc[idx] = drv_state
            dnf_hazard_driver.loc[idx] = dnf_drv_state

            tq = row.get("teammate_quali_gap_s")
            tr = row.get("teammate_race_gap")
            skill_parts = [tq, tr]
            skill_parts = [float(x) for x in skill_parts if pd.notna(x)]
            if skill_parts:
                skill_obs = float(np.mean(skill_parts))
                drv_state = DRIVER_EWMA_ALPHA * skill_obs + (1 - DRIVER_EWMA_ALPHA) * drv_state
            if pd.notna(row.get("dnf")):
                dnf_drv_state = DNF_EWMA_ALPHA * float(row["dnf"]) + (1 - DNF_EWMA_ALPHA) * dnf_drv_state

    out["car_pace_ewma"] = car_rating
    out["driver_skill_ewma"] = driver_rating
    out["car_driver_combo"] = out["car_pace_ewma"] + out["driver_skill_ewma"]
    out["dnf_hazard_car"] = dnf_hazard_car.clip(0.02, 0.85)
    out["dnf_hazard_driver"] = dnf_hazard_driver.clip(0.02, 0.85)
    out["dnf_hazard_combined"] = (
        1.0 - (1.0 - out["dnf_hazard_car"]) * (1.0 - out["dnf_hazard_driver"])
    ).clip(0.02, 0.90)
    return out
