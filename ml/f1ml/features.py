from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from f1ml.paths import CANONICAL, MANUAL, PROCESSED, RAW
from f1ml.ratings import build_ratings
from f1ml.safety_car import add_prior_circuit_safety_features

CLASSIFIED = ("Finished", "Lapped")


def _is_classified(status: str | None) -> bool:
    if not status:
        return False
    if status in CLASSIFIED:
        return True
    return status.startswith("+") and "Lap" in status


def _dnf_labels(status: pd.Series) -> pd.Series:
    """Return retirement/non-classification labels; exclude DNS, DSQ, and withdrawal."""
    normalized = status.fillna("").astype(str)
    labels = (~normalized.map(_is_classified)).astype(float)
    labels.loc[normalized.isin({"Did not start", "Disqualified", "Withdrew"})] = np.nan
    return labels


def _parse_pit_duration(value: str | float | None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(text)
    except ValueError:
        return None


def _merge_fp_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Attach FP1/FP2/FP3 best-lap and long-run delta vs field median."""
    path = CANONICAL / "fp_pace.parquet"
    for col in (
        "fp1_best_lap_delta",
        "fp2_best_lap_delta",
        "fp3_best_lap_delta",
        "fp_best_delta",
        "fp_longrun_delta",
        "race_pace_delta",
    ):
        df[col] = np.nan
    if not path.exists():
        return df

    fp = pd.read_parquet(path)
    session_map = {
        "FP1": ("fp1_best_lap_delta", "FP1_LONGRUN"),
        "FP2": ("fp2_best_lap_delta", "FP2_LONGRUN"),
        "FP3": ("fp3_best_lap_delta", "FP3_LONGRUN"),
        "Race": ("race_pace_delta", None),
    }
    longrun_candidates: dict[str, pd.Series] = {}
    for session, (best_col, long_col) in session_map.items():
        sub = fp[fp["session_type"] == session].copy()
        if sub.empty:
            continue
        medians = (
            sub.groupby(["season", "round"])["best_lap_seconds"]
            .median()
            .rename("session_median")
        )
        sub = sub.merge(medians, on=["season", "round"], how="left")
        sub[best_col] = sub["best_lap_seconds"] - sub["session_median"]
        if session == "Race":
            sub["race_pace_pct"] = 100.0 * sub[best_col] / sub["session_median"]
            df = df.drop(columns=["race_pace_pct"], errors="ignore")
            df = df.merge(
                sub[["season", "round", "driver_id", "race_pace_pct"]],
                on=["season", "round", "driver_id"],
                how="left",
            )
        merge_cols = sub[["season", "round", "driver_id", best_col]]
        df = df.drop(columns=[best_col], errors="ignore")
        df = df.merge(merge_cols, on=["season", "round", "driver_id"], how="left")

        if long_col:
            long_sub = fp[fp["session_type"] == long_col].copy()
            if not long_sub.empty:
                long_med = (
                    long_sub.groupby(["season", "round"])["best_lap_seconds"]
                    .median()
                    .rename("session_median")
                )
                long_sub = long_sub.merge(long_med, on=["season", "round"], how="left")
                long_sub["fp_longrun_delta"] = (
                    long_sub["best_lap_seconds"] - long_sub["session_median"]
                )
                lmerge = long_sub.set_index(["season", "round", "driver_id"])[
                    "fp_longrun_delta"
                ]
                longrun_candidates[session] = lmerge

    df["fp_best_delta"] = df["fp3_best_lap_delta"].fillna(df["fp2_best_lap_delta"]).fillna(
        df["fp1_best_lap_delta"]
    )
    keys = pd.MultiIndex.from_frame(df[["season", "round", "driver_id"]])
    df["fp_longrun_delta"] = np.nan
    for session in ("FP3", "FP2", "FP1"):
        values = longrun_candidates.get(session)
        if values is not None:
            df["fp_longrun_delta"] = df["fp_longrun_delta"].fillna(
                pd.Series(values.reindex(keys).to_numpy(), index=df.index)
            )
    df["fp_longrun_delta"] = df["fp_longrun_delta"].fillna(df["fp_best_delta"])
    return df


def _merge_pitstops(df: pd.DataFrame) -> pd.DataFrame:
    path = RAW / "pitstops.parquet"
    df = df.drop(columns=["pit_stops_avg", "pit_time_avg_s"], errors="ignore")
    if not path.exists():
        df["pit_stops_avg"] = np.nan
        df["pit_time_avg_s"] = np.nan
        return df
    pits = pd.read_parquet(path)
    if pits.empty:
        return df
    pits = pits.copy()
    pits["duration_s"] = pits["duration"].map(_parse_pit_duration)
    per_race = pits.groupby(["season", "round", "driver_id"], as_index=False).agg(
        race_pit_stops=("stop", "count"),
        race_pit_time_s=("duration_s", "mean"),
    )
    # Roll over every race the driver entered (not only races with pit data) so the
    # upcoming race gets the same prior-race value as training rows do.
    out = df.merge(per_race, on=["season", "round", "driver_id"], how="left")
    out = out.sort_values(["season", "round"])
    grp = out.groupby("driver_id", sort=False)
    out["pit_stops_avg"] = grp["race_pit_stops"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    out["pit_time_avg_s"] = grp["race_pit_time_s"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    return out.drop(columns=["race_pit_stops", "race_pit_time_s"]).sort_index()


def _merge_weather_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Overlay forecast precipitation for races not yet run."""
    path = CANONICAL / "weather_forecast.parquet"
    if not path.exists():
        return df
    fc = pd.read_parquet(path)
    if fc.empty:
        return df
    completed = df.groupby(["season", "round"])["position"].transform(lambda s: s.notna().any())
    upcoming_mask = ~completed
    if not upcoming_mask.any():
        return df
    fc_view = fc.rename(
        columns={
            "precipitation_mm": "forecast_precipitation_mm",
            "is_wet": "forecast_is_wet",
        }
    )
    df = df.merge(
        fc_view[["season", "round", "forecast_precipitation_mm", "forecast_is_wet"]],
        on=["season", "round"],
        how="left",
    )
    rain_mask = upcoming_mask & df["forecast_precipitation_mm"].notna()
    df.loc[rain_mask, "precipitation_mm"] = df.loc[rain_mask, "forecast_precipitation_mm"]
    wet_mask = upcoming_mask & df["forecast_is_wet"].notna()
    df["is_wet"] = df["is_wet"].astype(float)
    df.loc[wet_mask, "is_wet"] = df.loc[wet_mask, "forecast_is_wet"].astype(float)
    df = df.drop(columns=["forecast_precipitation_mm", "forecast_is_wet"], errors="ignore")
    return df


def _rolling_by_entity(df: pd.DataFrame, entity: str, windows: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """Shifted rolling stats so the current race is not included."""
    out = df.sort_values(["season", "round"]).copy()
    grp = out.groupby(entity, sort=False)
    for w in windows:
        out[f"{entity}_points_last_{w}"] = grp["points"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).sum()
        )
        out[f"{entity}_avg_finish_last_{w}"] = grp["position"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        out[f"{entity}_dnf_rate_last_{w}"] = grp["dnf"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    return out


def _add_circuit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit summaries using completed prior events only."""
    out = df.sort_values(["date", "season", "round", "driver_id"]).copy()
    out["grid_finish_delta"] = out["grid"] - out["position"]
    circuit_races = (
        out.groupby(["season", "round", "circuit_id"], as_index=False)
        .agg(race_grid_finish_delta=("grid_finish_delta", "mean"))
        .sort_values(["season", "round"])
    )
    circuit_races["circuit_grid_to_finish_delta"] = circuit_races.groupby(
        "circuit_id", sort=False
    )["race_grid_finish_delta"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    out = out.merge(
        circuit_races[["season", "round", "circuit_id", "circuit_grid_to_finish_delta"]],
        on=["season", "round", "circuit_id"],
        how="left",
    )
    out["circuit_avg_finish_prior"] = (
        out.sort_values(["date", "season", "round", "driver_id"])
        .groupby(["driver_id", "circuit_id"], sort=False)["position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return out


GRID_IMPORTANCE_PRIOR_RACES = 2.0


def load_circuit_profile() -> pd.DataFrame:
    """Manual track profile: speed class (high/medium/low) and street flag."""
    path = MANUAL / "circuit_profile.csv"
    if not path.exists():
        return pd.DataFrame(columns=["circuit_id", "speed_class", "street"])
    return pd.read_csv(path)[["circuit_id", "speed_class", "street"]]


def attach_circuit_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Attach speed_class plus one-hot columns the model can use."""
    out = df.drop(
        columns=["speed_class", "profile_street", "track_high_speed", "track_low_speed"],
        errors="ignore",
    )
    profile = load_circuit_profile().rename(columns={"street": "profile_street"})
    out = out.merge(profile, on="circuit_id", how="left")
    out["speed_class"] = out["speed_class"].fillna("medium")
    out["track_high_speed"] = (out["speed_class"] == "high").astype(float)
    out["track_low_speed"] = (out["speed_class"] == "low").astype(float)
    return out


def add_grid_importance(df: pd.DataFrame, prior_races: float = GRID_IMPORTANCE_PRIOR_RACES) -> pd.DataFrame:
    """How strongly the starting grid decided the result at this circuit, using prior races only.

    Per race: Spearman correlation between grid slot and finishing position
    (1.0 = finished exactly in grid order, e.g. Monaco). Per circuit: average of
    earlier races, shrunk toward the all-circuit average of earlier races so a
    track with one past race does not get an extreme value.
    """
    out = df.drop(columns=["circuit_grid_importance"], errors="ignore")
    rows = []
    for (season, rnd, circuit), g in out.groupby(["season", "round", "circuit_id"], sort=False):
        sub = g.dropna(subset=["grid", "position"])
        rho = np.nan
        if len(sub) >= 5 and sub["grid"].nunique() > 1:
            rho = float(spearmanr(sub["grid"], sub["position"]).correlation)
        rows.append({"season": season, "round": rnd, "circuit_id": circuit, "date": g["date"].iloc[0], "rho": rho})
    if not rows:
        out["circuit_grid_importance"] = np.nan
        return out
    races = pd.DataFrame(rows)
    races["_date"] = pd.to_datetime(races["date"], errors="coerce")
    races = races.sort_values(["_date", "season", "round"]).reset_index(drop=True)
    valid = races["rho"].notna()
    # All-circuit prior from strictly earlier races.
    global_sum = races["rho"].where(valid, 0.0).cumsum().shift(1).fillna(0.0)
    global_n = valid.astype(float).cumsum().shift(1).fillna(0.0)
    global_mean = (global_sum / global_n.replace(0, np.nan)).fillna(0.6)
    grp = races.groupby("circuit_id", sort=False)
    circ_sum = grp["rho"].transform(lambda s: s.fillna(0.0).cumsum().shift(1)).fillna(0.0)
    circ_n = grp["rho"].transform(lambda s: s.notna().astype(float).cumsum().shift(1)).fillna(0.0)
    races["circuit_grid_importance"] = (circ_sum + prior_races * global_mean) / (circ_n + prior_races)
    return out.merge(
        races[["season", "round", "circuit_id", "circuit_grid_importance"]],
        on=["season", "round", "circuit_id"],
        how="left",
    )


def impute_missing_quali_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gap features when the qualifying order is known but lap times are missing.

    Uses the median gap for that qualifying position in earlier races of the same
    regulation era (e.g. "P5 is typically 0.6% off pole this season").
    """
    out = df.copy()
    missing = out["quali_position"].notna() & out["best_quali_seconds"].isna()
    if not missing.any():
        return out
    era = (out["season"] >= 2026).astype(int)
    for (season, rnd), idx in out[missing].groupby(["season", "round"]).groups.items():
        race_era = int(season >= 2026)
        hist = out[
            (era == race_era)
            & out["best_quali_seconds"].notna()
            & ((out["season"] < season) | ((out["season"] == season) & (out["round"] < rnd)))
        ]
        if hist.empty:
            continue
        for col in ("quali_gap_pct", "quali_gap_s", "quali_gap_to_median_s"):
            med = hist.groupby("quali_position")[col].median()
            out.loc[idx, col] = out.loc[idx, "quali_position"].map(med).to_numpy()
        race = out.loc[idx]
        team_best = race.groupby("constructor_id")["quali_gap_pct"].transform("min")
        pole = hist["best_quali_seconds"].median()
        out.loc[idx, "teammate_quali_gap_pct"] = (race["quali_gap_pct"] - team_best).to_numpy()
        out.loc[idx, "teammate_quali_gap_s"] = (
            (race["quali_gap_pct"] - team_best) * pole / 100.0
        ).to_numpy()
    return out


def add_track_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Grid/quali effects scaled by how much track position matters at this circuit."""
    out = df.copy()
    gi = out.get("circuit_grid_importance", pd.Series(np.nan, index=out.index))
    out["grid_x_grid_importance"] = out.get("grid", np.nan) * gi
    out["quali_gap_x_grid_importance"] = out.get("quali_gap_pct", np.nan) * gi
    return out


def _append_upcoming_race(
    results: pd.DataFrame, races: pd.DataFrame, quali: pd.DataFrame
) -> pd.DataFrame:
    """Add empty result rows for the next race so it flows through the same feature code.

    Entry list comes from that race's qualifying when available, otherwise from
    the most recent completed race. Result columns stay empty (NaN).
    """
    done = set(zip(results["season"], results["round"]))
    cal = races.copy()
    cal["_date"] = pd.to_datetime(cal["date"], errors="coerce")
    pending = cal[[(s, r) not in done for s, r in zip(cal["season"], cal["round"])]]
    if pending.empty or results.empty:
        return results.assign(is_upcoming=0)
    nxt = pending.sort_values("_date").iloc[0]
    season, rnd = int(nxt["season"]), int(nxt["round"])
    entry_cols = [
        "driver_id", "driver_code", "driver_number", "given_name", "family_name",
        "nationality", "constructor_id", "constructor_name", "constructor_nationality",
    ]
    q = quali[(quali["season"] == season) & (quali["round"] == rnd)]
    if not q.empty:
        entry = q[entry_cols].drop_duplicates("driver_id")
    else:
        last = results.sort_values(["season", "round"]).iloc[-1]
        prev = results[(results["season"] == last["season"]) & (results["round"] == last["round"])]
        entry = prev[entry_cols].drop_duplicates("driver_id")
    upcoming = entry.copy()
    upcoming["season"] = season
    upcoming["round"] = rnd
    for col in ("position", "points", "grid", "laps"):
        upcoming[col] = np.nan
    for col in ("position_text", "status", "time"):
        upcoming[col] = None
    out = pd.concat([results.assign(is_upcoming=0), upcoming.assign(is_upcoming=1)], ignore_index=True)
    return out


def _apply_upcoming_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Use the published starting grid (or qualifying order) for the upcoming race."""
    mask = df["is_upcoming"] == 1
    if not mask.any():
        return df
    from f1ml.starting_grid import apply_grid_to_frame

    sub = df[mask]
    season, rnd = int(sub["season"].iloc[0]), int(sub["round"].iloc[0])
    try:
        gridded = apply_grid_to_frame(sub.copy(), season, rnd)
    except Exception:
        gridded = sub.copy()
        gridded["grid"] = gridded["quali_position"]
    df.loc[mask, "grid"] = gridded["grid"].to_numpy()
    return df


def attach_circuit_type_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a sourced street-circuit flag where OpenF1 meeting data exists."""
    out = df.drop(columns=["circuit_type", "circuit_street", "car_street_interaction"], errors="ignore").copy()
    path = RAW / "openf1_race_sessions.parquet"
    if path.exists():
        metadata = pd.read_parquet(path)
        if {"season", "round", "circuit_type"}.issubset(metadata.columns):
            metadata = metadata[["season", "round", "circuit_type"]].drop_duplicates(
                ["season", "round"]
            )
            out = out.merge(metadata, on=["season", "round"], how="left")
        else:
            out["circuit_type"] = np.nan
    else:
        out["circuit_type"] = np.nan
    street = out["circuit_type"].astype("string").str.contains("street", case=False, na=pd.NA)
    out["circuit_street"] = street.astype("Float64").astype(float)
    if "profile_street" in out.columns:
        # Manual profile covers every circuit; OpenF1 metadata only 2023+.
        out["circuit_street"] = out["profile_street"].astype(float).fillna(out["circuit_street"])
    out["car_street_interaction"] = out.get("car_pace_ewma", np.nan) * out["circuit_street"]
    return out


def build_driver_race() -> pd.DataFrame:
    races = pd.read_parquet(RAW / "races.parquet")
    results = pd.read_parquet(RAW / "results.parquet")
    quali = pd.read_parquet(RAW / "qualifying.parquet")
    sprints = pd.read_parquet(RAW / "sprints.parquet")
    standings = pd.read_parquet(RAW / "standings.parquet")
    weather = pd.read_parquet(RAW / "weather.parquet")

    results = _append_upcoming_race(results, races, quali)
    df = results.merge(
        races[
            [
                "season",
                "round",
                "race_name",
                "date",
                "circuit_id",
                "circuit_name",
                "country",
            ]
        ],
        on=["season", "round"],
        how="left",
    )
    df = df.merge(
        quali[
            [
                "season",
                "round",
                "driver_id",
                "quali_position",
                "best_quali_seconds",
                "q3_seconds",
            ]
        ],
        on=["season", "round", "driver_id"],
        how="left",
    )
    if len(sprints):
        df = df.merge(
            sprints[["season", "round", "driver_id", "sprint_position", "sprint_points"]],
            on=["season", "round", "driver_id"],
            how="left",
        )
    else:
        df["sprint_position"] = np.nan
        df["sprint_points"] = np.nan

    df = df.merge(weather, on=["season", "round"], how="left")

    prev = standings.rename(
        columns={
            "after_round": "prev_round",
            "standing_position": "champ_position_before",
            "points": "champ_points_before",
            "wins": "champ_wins_before",
        }
    )
    df["prev_round"] = df["round"] - 1
    df = df.merge(
        prev[
            [
                "season",
                "prev_round",
                "driver_id",
                "champ_position_before",
                "champ_points_before",
                "champ_wins_before",
            ]
        ],
        on=["season", "prev_round", "driver_id"],
        how="left",
    )
    df.loc[df["round"] == 1, ["champ_position_before", "champ_points_before", "champ_wins_before"]] = [
        np.nan,
        0.0,
        0.0,
    ]

    df["dnf"] = _dnf_labels(df["status"])
    df.loc[df["is_upcoming"] == 1, "dnf"] = np.nan
    df["won"] = (df["position"] == 1).astype(int)
    df["points_finish"] = (df["position"] <= 10).astype(int)
    df["podium"] = df["position"].isin([1, 2, 3]).astype(int)
    df["has_sprint"] = df["sprint_position"].notna().astype(int)
    df["grid"] = df["grid"].replace(0, np.nan)
    df = _apply_upcoming_grid(df)
    df["grid_vs_quali"] = df["grid"] - df["quali_position"]

    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    df = _rolling_by_entity(df, "driver_id")

    df = df.sort_values(["season", "round", "constructor_id"]).reset_index(drop=True)
    grouped = df.groupby(["season", "round", "constructor_id"], as_index=False).agg(
        constructor_race_points=("points", "sum"),
        constructor_race_dnf=("dnf", "mean"),
        constructor_best_finish=("position", "min"),
    )
    grouped = grouped.sort_values(["season", "round"])
    cgrp = grouped.groupby("constructor_id", sort=False)
    for w in (3, 5):
        grouped[f"constructor_id_points_last_{w}"] = cgrp["constructor_race_points"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).sum()
        )
        grouped[f"constructor_id_avg_finish_last_{w}"] = cgrp["constructor_best_finish"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        grouped[f"constructor_id_dnf_rate_last_{w}"] = cgrp["constructor_race_dnf"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    df = df.merge(
        grouped[
            [
                "season",
                "round",
                "constructor_id",
                "constructor_id_points_last_3",
                "constructor_id_points_last_5",
                "constructor_id_avg_finish_last_3",
                "constructor_id_avg_finish_last_5",
                "constructor_id_dnf_rate_last_5",
            ]
        ],
        on=["season", "round", "constructor_id"],
        how="left",
    )

    teammate_quali = df.groupby(["season", "round", "constructor_id"])["quali_position"].transform("min")
    df["quali_vs_teammate"] = df["quali_position"] - teammate_quali

    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    # quali_vs_teammate is 0 for the faster driver of a pairing (gap to team best).
    team_quali_n = df.groupby(["season", "round", "constructor_id"])["quali_position"].transform("count")
    df["beat_teammate_quali"] = (df["quali_vs_teammate"] == 0).astype(float)
    df.loc[df["quali_position"].isna() | (team_quali_n < 2), "beat_teammate_quali"] = np.nan
    df["teammate_quali_h2h_rate"] = (
        df.groupby("driver_id", sort=False)["beat_teammate_quali"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df = _add_circuit_history_features(df)
    df = attach_circuit_profile(df)
    df = add_grid_importance(df)
    safety_labels_path = RAW / "openf1_safety_car_labels.parquet"
    if safety_labels_path.exists():
        safety_labels = pd.read_parquet(safety_labels_path)
        df = add_prior_circuit_safety_features(df, safety_labels)
    else:
        df["circuit_sc_rate_prior"] = np.nan
        df["circuit_vsc_rate_prior"] = np.nan

    df = _merge_fp_pace(df)
    df = _merge_pitstops(df)
    df = _merge_weather_forecast(df)

    df["is_wet"] = df["is_wet"].fillna(False).astype(int)
    df["sprint_position"] = df["sprint_position"].fillna(-1)

    # Ratings + pace gaps (shifted car/driver EWMA, quali_gap_s, etc.).
    df = build_ratings(df)
    df = attach_circuit_type_feature(df)
    df = impute_missing_quali_gaps(df)
    df = add_track_interactions(df)

    numeric_fill = [
        "quali_position",
        "quali_gap_s",
        "quali_gap_to_median_s",
        "grid",
        "best_quali_seconds",
        "q3_seconds",
        "champ_position_before",
        "champ_points_before",
        "champ_wins_before",
        "driver_id_points_last_3",
        "driver_id_points_last_5",
        "driver_id_avg_finish_last_5",
        "driver_id_dnf_rate_last_5",
        "constructor_id_points_last_3",
        "constructor_id_points_last_5",
        "constructor_id_avg_finish_last_5",
        "constructor_id_dnf_rate_last_5",
        "car_pace_ewma",
        "driver_skill_ewma",
        "car_driver_combo",
        "dnf_hazard_car",
        "dnf_hazard_driver",
        "dnf_hazard_combined",
        "quali_vs_teammate",
        "teammate_quali_gap_s",
        "grid_vs_quali",
        "circuit_avg_finish_prior",
        "circuit_grid_to_finish_delta",
        "teammate_quali_h2h_rate",
        "precipitation_mm",
        "fp1_best_lap_delta",
        "fp2_best_lap_delta",
        "fp3_best_lap_delta",
        "fp_best_delta",
        "fp_longrun_delta",
        "race_pace_delta",
        "pit_stops_avg",
        "pit_time_avg_s",
        "regs_era",
        "circuit_street",
        "car_street_interaction",
        "sprint_position",
        "circuit_grid_importance",
        "grid_x_grid_importance",
        "quali_gap_x_grid_importance",
        "quali_gap_pct",
        "car_quali_class_ewma",
        "car_class_delta",
        "driver_race_vs_teammate_ewma",
        "car_race_pace_ewma",
    ]
    for col in numeric_fill:
        if col in df.columns:
            df[col] = df[col].astype(float)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "driver_race.parquet", index=False)
    df.to_csv(PROCESSED / "driver_race.csv", index=False)
    return df
