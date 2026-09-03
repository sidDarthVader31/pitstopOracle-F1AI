from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from f1ml.paths import MODELS, PROCESSED, RAW, REPORTS

ModelKind = Literal["rf", "logreg"]
RaceMode = Literal["complete", "post_quali", "pre_quali"]


@lru_cache(maxsize=1)
def load_driver_race() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "driver_race.parquet")


@lru_cache(maxsize=1)
def load_feature_spec() -> dict:
    return joblib.load(MODELS / "features.joblib")


@lru_cache(maxsize=2)
def load_model(kind: ModelKind) -> Pipeline:
    path = MODELS / ("winner_rf.joblib" if kind == "rf" else "winner_logreg.joblib")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    return json.loads((REPORTS / "metrics.json").read_text())


def list_races(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per completed race weekend (has results in feature table)."""
    df = df if df is not None else load_driver_race()
    races = (
        df.groupby(["season", "round"], as_index=False)
        .agg(
            race_name=("race_name", "first"),
            date=("date", "first"),
            circuit_name=("circuit_name", "first"),
            country=("country", "first"),
            has_winner=("won", "max"),
            n_drivers=("driver_id", "count"),
        )
        .sort_values(["season", "round"])
    )
    races["label"] = races.apply(
        lambda r: f"{int(r['season'])} R{int(r['round']):02d} — {r['race_name']}", axis=1
    )
    return races


@lru_cache(maxsize=1)
def list_calendar_races() -> pd.DataFrame:
    """Full season calendar merged with completion / quali status."""
    calendar = pd.read_parquet(RAW / "races.parquet")
    history = load_driver_race()
    status = (
        history.groupby(["season", "round"], as_index=False)
        .agg(
            has_winner=("won", "max"),
            has_quali=("quali_position", lambda s: s.notna().any()),
            n_drivers=("driver_id", "count"),
        )
    )
    races = calendar.merge(status, on=["season", "round"], how="left")
    races["has_winner"] = races["has_winner"].fillna(0).astype(int)
    races["has_quali"] = races["has_quali"].fillna(False).astype(bool)
    races["n_drivers"] = races["n_drivers"].fillna(0).astype(int)
    races["status"] = races.apply(_race_status_row, axis=1)
    races["label"] = races.apply(
        lambda r: f"{int(r['season'])} R{int(r['round']):02d} — {r['race_name']}", axis=1
    )
    return races.sort_values(["season", "round"]).reset_index(drop=True)


def _race_status_row(row: pd.Series) -> RaceMode:
    if int(row.get("has_winner", 0)) == 1:
        return "complete"
    if bool(row.get("has_quali", False)):
        return "post_quali"
    return "pre_quali"


def race_status(race_df: pd.DataFrame) -> RaceMode:
    if race_df["won"].max() == 1:
        return "complete"
    if race_df["quali_position"].notna().any():
        return "post_quali"
    return "pre_quali"


def status_label(mode: RaceMode) -> str:
    return {
        "complete": "Race complete",
        "post_quali": "Post-qualifying",
        "pre_quali": "Pre-qualifying",
    }[mode]


def next_upcoming_race() -> tuple[int, int]:
    """First calendar race on or after today (latest season if multiple)."""
    today = date.today()
    cal = list_calendar_races()
    cal["race_date"] = pd.to_datetime(cal["date"]).dt.date
    upcoming = cal[cal["race_date"] >= today]
    if upcoming.empty:
        row = cal.iloc[-1]
    else:
        row = upcoming.iloc[0]
    return int(row["season"]), int(row["round"])


def resolve_race_features(season: int, round_num: int) -> tuple[pd.DataFrame, RaceMode]:
    """Load features for completed, post-quali, or pre-weekend races."""
    try:
        race_df = get_race_features(season, round_num)
        return race_df, race_status(race_df)
    except ValueError:
        race_df = build_upcoming_race_features(season, round_num)
        return race_df, race_status(race_df)


def latest_race_with_quali(df: pd.DataFrame | None = None) -> tuple[int, int]:
    df = df if df is not None else load_driver_race()
    subset = df[df["quali_position"].notna()]
    row = subset.sort_values(["season", "round"]).iloc[-1]
    return int(row["season"]), int(row["round"])


def get_race_features(season: int, round_num: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else load_driver_race()
    race = df[(df["season"] == season) & (df["round"] == round_num)].copy()
    if race.empty:
        raise ValueError(f"No data for season={season} round={round_num}")
    return race.reset_index(drop=True)


def _latest_completed_round(season: int, before_round: int, history: pd.DataFrame) -> int:
    """Latest round in season with results, strictly before the target round."""
    completed = history[(history["season"] == season) & (history["round"] < before_round)]
    if completed.empty:
        raise ValueError(f"No completed races in {season} before round {before_round}")
    return int(completed["round"].max())


def build_upcoming_race_features(season: int, round_num: int) -> pd.DataFrame:
    """Build pre-race feature rows before qualifying/results exist."""
    from f1ml.paths import RAW

    history = load_driver_race()
    calendar = pd.read_parquet(RAW / "races.parquet")
    standings = pd.read_parquet(RAW / "standings.parquet")
    race_meta = calendar[(calendar["season"] == season) & (calendar["round"] == round_num)]
    if race_meta.empty:
        raise ValueError(f"Round {round_num} not on {season} calendar")
    meta = race_meta.iloc[0]
    prev_round = _latest_completed_round(season, round_num, history)

    prev = get_race_features(season, prev_round, history)

    season_hist = history[history["season"] == season].sort_values("round")
    prior = season_hist[season_hist["round"] < round_num]

    out = prev.copy()
    out["round"] = round_num
    out["race_name"] = meta["race_name"]
    out["date"] = meta["date"]
    out["circuit_id"] = meta["circuit_id"]
    out["circuit_name"] = meta["circuit_name"]
    out["country"] = meta["country"]

    champ = standings[(standings["season"] == season) & (standings["after_round"] == prev_round)]
    champ = champ.rename(
        columns={
            "standing_position": "champ_position_before",
            "points": "champ_points_before",
            "wins": "champ_wins_before",
        }
    )
    out = out.drop(columns=["champ_position_before", "champ_points_before", "champ_wins_before"], errors="ignore")
    out = out.merge(
        champ[["driver_id", "champ_position_before", "champ_points_before", "champ_wins_before"]],
        on="driver_id",
        how="left",
    )

    for w in (3, 5):
        out[f"driver_id_points_last_{w}"] = (
            prior.groupby("driver_id")["points"]
            .apply(lambda s: s.tail(w).sum())
            .reindex(out["driver_id"])
            .values
        )
        out[f"driver_id_avg_finish_last_{w}"] = (
            prior.groupby("driver_id")["position"]
            .apply(lambda s: s.tail(w).mean())
            .reindex(out["driver_id"])
            .values
        )
        out[f"driver_id_dnf_rate_last_{w}"] = (
            prior.groupby("driver_id")["dnf"]
            .apply(lambda s: s.tail(w).mean())
            .reindex(out["driver_id"])
            .values
        )

    team_hist = (
        prior.groupby(["round", "constructor_id"], as_index=False)
        .agg(team_pts=("points", "sum"), team_best=("position", "min"), team_dnf=("dnf", "mean"))
        .sort_values("round")
    )
    for w in (3, 5):
        out[f"constructor_id_points_last_{w}"] = (
            team_hist.groupby("constructor_id")["team_pts"]
            .apply(lambda s: s.tail(w).sum())
            .reindex(out["constructor_id"])
            .values
        )
        out[f"constructor_id_avg_finish_last_{w}"] = (
            team_hist.groupby("constructor_id")["team_best"]
            .apply(lambda s: s.tail(w).mean())
            .reindex(out["constructor_id"])
            .values
        )
        out[f"constructor_id_dnf_rate_last_{w}"] = (
            team_hist.groupby("constructor_id")["team_dnf"]
            .apply(lambda s: s.tail(w).mean())
            .reindex(out["constructor_id"])
            .values
        )

    circuit_id = meta["circuit_id"]
    monza_hist = history[history["circuit_id"] == circuit_id]
    out["circuit_avg_finish_prior"] = (
        monza_hist.groupby("driver_id")["position"]
        .mean()
        .reindex(out["driver_id"])
        .values
    )

    for col in ("quali_position", "grid", "grid_vs_quali", "best_quali_seconds", "q3_seconds"):
        out[col] = np.nan
    out["has_sprint"] = 0
    out["sprint_position"] = -1
    out["sprint_points"] = np.nan
    out["is_wet"] = 0
    out["precipitation_mm"] = 0.0
    out["quali_vs_teammate"] = np.nan
    out["position"] = np.nan
    out["won"] = 0
    out["podium"] = 0
    out["dnf"] = 0

    return out.reset_index(drop=True)


def apply_what_if(
    race_df: pd.DataFrame,
    *,
    is_wet: bool | None = None,
    grid_overrides: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Adjust pre-race features for scenario toggles without retraining."""
    out = race_df.copy()
    if is_wet is not None:
        out["is_wet"] = int(is_wet)
        if is_wet:
            out["precipitation_mm"] = out["precipitation_mm"].fillna(5.0).clip(lower=1.0)
        else:
            out["precipitation_mm"] = 0.0

    if grid_overrides:
        for driver_id, grid_pos in grid_overrides.items():
            mask = out["driver_id"] == driver_id
            if not mask.any():
                continue
            out.loc[mask, "grid"] = float(grid_pos)
            quali = out.loc[mask, "quali_position"].iloc[0]
            if pd.notna(quali):
                out.loc[mask, "grid_vs_quali"] = float(grid_pos) - float(quali)
            else:
                out.loc[mask, "quali_position"] = float(grid_pos)
                out.loc[mask, "grid_vs_quali"] = 0.0

    return out


def _feature_matrix(race_df: pd.DataFrame) -> pd.DataFrame:
    spec = load_feature_spec()
    cols = spec["numeric"] + spec["categorical"]
    return race_df[cols]


def score_race(
    race_df: pd.DataFrame,
    kind: ModelKind = "rf",
    normalize: bool = True,
) -> pd.DataFrame:
    """Return race rows with raw and optionally normalized win probabilities."""
    model = load_model(kind)
    X = _feature_matrix(race_df)
    proba = model.predict_proba(X)[:, 1]
    out = race_df.copy()
    out["win_prob_raw"] = proba
    if normalize and proba.sum() > 0:
        out["win_prob"] = proba / proba.sum()
    else:
        out["win_prob"] = proba
    out["model_pick"] = out["win_prob"] == out["win_prob"].max()
    return out.sort_values("win_prob", ascending=False).reset_index(drop=True)


def pole_pick(race_df: pd.DataFrame) -> pd.Series:
    valid = race_df.dropna(subset=["quali_position"])
    if valid.empty:
        raise ValueError("No qualifying data")
    idx = valid["quali_position"].idxmin()
    row = race_df.loc[idx]
    return pd.Series(
        {
            "driver_id": row["driver_id"],
            "driver_name": f"{row['given_name']} {row['family_name']}",
            "quali_position": row["quali_position"],
        }
    )


def safe_pole_pick(race_df: pd.DataFrame) -> pd.Series | None:
    try:
        return pole_pick(race_df)
    except ValueError:
        return None


def champ_leader_pick(race_df: pd.DataFrame) -> pd.Series:
    subset = race_df.dropna(subset=["champ_position_before"])
    if subset.empty:
        row = race_df.iloc[0]
    else:
        row = subset.loc[subset["champ_position_before"].idxmin()]
    return pd.Series(
        {
            "driver_id": row["driver_id"],
            "driver_name": f"{row['given_name']} {row['family_name']}",
            "champ_position": row.get("champ_position_before"),
            "champ_points": row.get("champ_points_before"),
        }
    )


def model_pick(scored: pd.DataFrame) -> pd.Series:
    row = scored.iloc[0]
    return pd.Series(
        {
            "driver_id": row["driver_id"],
            "driver_name": f"{row['given_name']} {row['family_name']}",
            "win_prob": row["win_prob"],
        }
    )


def display_columns(scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "given_name",
        "family_name",
        "constructor_name",
        "quali_position",
        "grid",
        "win_prob",
        "win_prob_raw",
        "champ_position_before",
        "is_wet",
    ]
    view = scored[cols].copy()
    view["win_prob"] = (view["win_prob"] * 100).round(1)
    view["win_prob_raw"] = (view["win_prob_raw"] * 100).round(1)
    view = view.rename(
        columns={
            "given_name": "First",
            "family_name": "Last",
            "constructor_name": "Team",
            "quali_position": "Quali",
            "grid": "Grid",
            "win_prob": "Win % (norm)",
            "win_prob_raw": "Win % (raw)",
            "champ_position_before": "Champ pos",
            "is_wet": "Wet",
        }
    )
    return view
