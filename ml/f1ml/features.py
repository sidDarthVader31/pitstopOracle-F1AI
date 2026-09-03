from __future__ import annotations

import numpy as np
import pandas as pd

from f1ml.paths import PROCESSED, RAW

CLASSIFIED = ("Finished",)
DNF_PREFIX_OK = ("+",)  # +1 Lap, +2 Laps, ...


def _is_classified(status: str | None) -> bool:
    if not status:
        return False
    if status in CLASSIFIED:
        return True
    return status.startswith("+") and "Lap" in status


def _rolling_by_entity(df: pd.DataFrame, entity: str, windows: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """Shifted rolling stats so the current race is not included."""
    out = df.sort_values(["season", "round"]).copy()
    grouped = out.groupby(entity, sort=False)
    shifted_points = grouped["points"].shift(1)
    shifted_finish = grouped["position"].shift(1)
    shifted_dnf = grouped["dnf"].shift(1)
    for w in windows:
        out[f"{entity}_points_last_{w}"] = shifted_points.groupby(out[entity]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        out[f"{entity}_avg_finish_last_{w}"] = shifted_finish.groupby(out[entity]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{entity}_dnf_rate_last_{w}"] = shifted_dnf.groupby(out[entity]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
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
        grouped[f"constructor_id_points_last_{w}"] = (
            cgrp["constructor_race_points"].shift(1).groupby(grouped["constructor_id"]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        grouped[f"constructor_id_avg_finish_last_{w}"] = (
            cgrp["constructor_best_finish"].shift(1).groupby(grouped["constructor_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        grouped[f"constructor_id_dnf_rate_last_{w}"] = (
            cgrp["constructor_race_dnf"].shift(1).groupby(grouped["constructor_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
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

    # Prior results at this circuit (exclude current race).
    df = df.sort_values(["date", "season", "round"])
    hist = []
    for (driver, circuit), g in df.groupby(["driver_id", "circuit_id"], sort=False):
        prev_finish = g["position"].shift(1)
        hist.append(prev_finish.expanding(min_periods=1).mean())
    df["circuit_avg_finish_prior"] = pd.concat(hist).sort_index()

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
        "precipitation_mm",
    ]
    for col in numeric_fill:
        if col in df.columns:
            df[col] = df[col].astype(float)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "driver_race.parquet", index=False)
    df.to_csv(PROCESSED / "driver_race.csv", index=False)
    return df
