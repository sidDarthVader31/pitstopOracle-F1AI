from __future__ import annotations

import numpy as np
import pandas as pd

from f1ml.paths import PROCESSED, RAW, CANONICAL

CLASSIFIED = ("Finished",)
DNF_PREFIX_OK = ("+",)  # +1 Lap, +2 Laps, ...


def _merge_fp_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Attach FP2/FP3 best-lap delta vs field median from canonical store."""
    path = CANONICAL / "fp_pace.parquet"
    df["fp2_best_lap_delta"] = np.nan
    df["fp3_best_lap_delta"] = np.nan
    if not path.exists():
        return df
    fp = pd.read_parquet(path)
    for session, col in (("FP2", "fp2_best_lap_delta"), ("FP3", "fp3_best_lap_delta")):
        sub = fp[fp["session_type"] == session].copy()
        if sub.empty:
            continue
        medians = (
            sub.groupby(["season", "round"])["best_lap_seconds"]
            .median()
            .rename("session_median")
        )
        sub = sub.merge(medians, on=["season", "round"], how="left")
        sub[col] = sub["best_lap_seconds"] - sub["session_median"]
        merge_cols = sub[["season", "round", "driver_id", col]]
        df = df.drop(columns=[col], errors="ignore")
        df = df.merge(merge_cols, on=["season", "round", "driver_id"], how="left")
    return df


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
    df.loc[upcoming_mask & df["forecast_precipitation_mm"].notna(), "precipitation_mm"] = df[
        "forecast_precipitation_mm"
    ]
    df.loc[upcoming_mask & df["forecast_is_wet"].notna(), "is_wet"] = df["forecast_is_wet"].astype(int)
    df = df.drop(columns=["forecast_precipitation_mm", "forecast_is_wet"], errors="ignore")
    return df


def _is_classified(status: str | None) -> bool:
    if not status:
        return False
    if status in CLASSIFIED:
        return True
    return status.startswith("+") and "Lap" in status


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


def build_driver_race() -> pd.DataFrame:
    races = pd.read_parquet(RAW / "races.parquet")
    results = pd.read_parquet(RAW / "results.parquet")
    quali = pd.read_parquet(RAW / "qualifying.parquet")
    sprints = pd.read_parquet(RAW / "sprints.parquet")
    standings = pd.read_parquet(RAW / "standings.parquet")
    weather = pd.read_parquet(RAW / "weather.parquet")

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

    # Pre-race standings = standings after previous round in the same season.
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

    df["dnf"] = (~df["status"].map(_is_classified)).astype(int)
    df["won"] = (df["position"] == 1).astype(int)
    df["points_finish"] = (df["position"] <= 10).astype(int)
    df["podium"] = df["position"].isin([1, 2, 3]).astype(int)
    df["has_sprint"] = df["sprint_position"].notna().astype(int)
    df["grid"] = df["grid"].replace(0, np.nan)  # 0 = pit lane / did not start from grid in Ergast
    df["grid_vs_quali"] = df["grid"] - df["quali_position"]

    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    df = _rolling_by_entity(df, "driver_id")
    # Constructor rolling uses same race rows; recompute on constructor_id
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

    teammate_quali = (
        df.groupby(["season", "round", "constructor_id"])["quali_position"].transform("min")
    )
    df["quali_vs_teammate"] = df["quali_position"] - teammate_quali

    # Teammate qualifying H2H win rate (shifted rolling).
    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    df["beat_teammate_quali"] = (
        df["quali_vs_teammate"] < 0
    ).astype(float)
    df.loc[df["quali_position"].isna(), "beat_teammate_quali"] = np.nan
    df["teammate_quali_h2h_rate"] = (
        df.groupby("driver_id", sort=False)["beat_teammate_quali"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Circuit overtaking proxy: historical mean (grid - finish) at circuit.
    df = df.sort_values(["date", "season", "round"])
    df["grid_finish_delta"] = df["grid"] - df["position"]
    circuit_overtake = (
        df.groupby("circuit_id")["grid_finish_delta"]
        .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )
    df["circuit_grid_to_finish_delta"] = circuit_overtake.fillna(0.0)

    # Prior results at this circuit (exclude current race).
    df = df.sort_values(["date", "season", "round"])
    df["circuit_avg_finish_prior"] = (
        df.groupby(["driver_id", "circuit_id"], sort=False)["position"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    df = _merge_fp_pace(df)
    df = _merge_weather_forecast(df)

    df["is_wet"] = df["is_wet"].fillna(False).astype(int)
    df["sprint_position"] = df["sprint_position"].fillna(-1)
    numeric_fill = [
        "quali_position",
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
        "quali_vs_teammate",
        "grid_vs_quali",
        "circuit_avg_finish_prior",
        "circuit_grid_to_finish_delta",
        "teammate_quali_h2h_rate",
        "precipitation_mm",
        "fp2_best_lap_delta",
        "fp3_best_lap_delta",
    ]
    for col in numeric_fill:
        if col in df.columns:
            df[col] = df[col].astype(float)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "driver_race.parquet", index=False)
    df.to_csv(PROCESSED / "driver_race.csv", index=False)
    return df
