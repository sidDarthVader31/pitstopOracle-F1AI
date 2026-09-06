from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from f1ml.fantasy import constructor_fantasy_points, expected_driver_fantasy_points, oracle_xi
from f1ml.modeling import WeekendModels, apply_race_probs, feature_matrix, predict_binary_proba, raw_win_scores
from f1ml.paths import MODELS, PROCESSED, RAW, REPORTS
from f1ml.specs import WeekendMode

ModelKind = Literal["hgb", "logreg", "lgbm", "rf"]  # lgbm/rf aliases for hgb
RaceMode = Literal["complete", "post_quali", "pre_quali"]


@lru_cache(maxsize=1)
def load_driver_race() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "driver_race.parquet")


@lru_cache(maxsize=2)
def load_weekend_models(mode: WeekendMode) -> WeekendModels:
    return WeekendModels.load(MODELS / mode)


@lru_cache(maxsize=1)
def load_feature_spec() -> dict:
    path = MODELS / "pre_quali" / "features.joblib"
    if path.exists():
        return joblib.load(path)
    return joblib.load(MODELS / "features.joblib")


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    return json.loads((REPORTS / "metrics.json").read_text())


def inference_mode(race_mode: RaceMode) -> WeekendMode:
    """Map UI race status to model bundle."""
    if race_mode in ("post_quali", "complete"):
        return "post_quali"
    return "pre_quali"


@lru_cache(maxsize=2)
def load_model(kind: ModelKind = "hgb") -> object:
    """Backward-compatible loader."""
    bundle = load_weekend_models("post_quali")
    if kind == "logreg" and bundle.win_logreg is not None:
        return bundle.win_logreg
    return bundle.win_model


def list_races(df: pd.DataFrame | None = None) -> pd.DataFrame:
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
    raw_quali = _raw_quali_status()
    if not raw_quali.empty:
        races = races.merge(raw_quali, on=["season", "round"], how="left", suffixes=("", "_raw"))
        races["has_quali"] = races["has_quali"].fillna(False) | races["has_quali_raw"].fillna(False)
        races = races.drop(columns=["has_quali_raw"], errors="ignore")
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
    try:
        race_df = get_race_features(season, round_num)
        return race_df, race_status(race_df)
    except ValueError:
        race_df = build_upcoming_race_features(season, round_num)
        return race_df, race_status(race_df)


def get_race_features(season: int, round_num: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else load_driver_race()
    race = df[(df["season"] == season) & (df["round"] == round_num)].copy()
    if race.empty:
        raise ValueError(f"No data for season={season} round={round_num}")
    return race.reset_index(drop=True)


def _latest_completed_round(season: int, before_round: int, history: pd.DataFrame) -> int:
    completed = history[(history["season"] == season) & (history["round"] < before_round)]
    if completed.empty:
        raise ValueError(f"No completed races in {season} before round {before_round}")
    return int(completed["round"].max())


def _raw_quali_status() -> pd.DataFrame:
    path = RAW / "qualifying.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["season", "round", "has_quali"])
    quali = pd.read_parquet(path)
    return (
        quali.groupby(["season", "round"], as_index=False)
        .agg(has_quali=("quali_position", lambda s: s.notna().any()))
    )


def _raw_quali_for_round(season: int, round_num: int) -> pd.DataFrame:
    path = RAW / "qualifying.parquet"
    if not path.exists():
        return pd.DataFrame()
    quali = pd.read_parquet(path)
    q = quali[(quali["season"] == season) & (quali["round"] == round_num)].copy()
    return q[q["quali_position"].notna()]


def _raw_sprint_for_round(season: int, round_num: int) -> pd.DataFrame:
    path = RAW / "sprints.parquet"
    if not path.exists():
        return pd.DataFrame()
    sprints = pd.read_parquet(path)
    return sprints[(sprints["season"] == season) & (sprints["round"] == round_num)].copy()


def _apply_upcoming_session_data(
    out: pd.DataFrame,
    season: int,
    round_num: int,
    meta: pd.Series,
    champ: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay qualifying and sprint from raw parquet before race results exist."""
    q = _raw_quali_for_round(season, round_num)
    if q.empty:
        for col in (
            "quali_position", "grid", "grid_vs_quali", "best_quali_seconds", "q3_seconds",
            "quali_vs_teammate",
        ):
            out[col] = np.nan
        out["has_sprint"] = 0
        out["sprint_position"] = -1
        out["sprint_points"] = np.nan
        return out

    missing_ids = set(q["driver_id"]) - set(out["driver_id"])
    if missing_ids:
        extra = q[q["driver_id"].isin(missing_ids)].copy()
        extra["season"] = season
        extra["round"] = round_num
        extra["race_name"] = meta["race_name"]
        extra["date"] = meta["date"]
        extra["circuit_id"] = meta["circuit_id"]
        extra["circuit_name"] = meta["circuit_name"]
        extra["country"] = meta["country"]
        extra = extra.merge(
            champ[["driver_id", "champ_position_before", "champ_points_before", "champ_wins_before"]],
            on="driver_id",
            how="left",
        )
        extra["position"] = np.nan
        extra["won"] = 0
        extra["podium"] = 0
        extra["dnf"] = 0
        extra["is_wet"] = 0
        extra["precipitation_mm"] = 0.0
        out = pd.concat([out, extra], ignore_index=True)

    q_merge = q[
        ["driver_id", "quali_position", "best_quali_seconds", "q3_seconds"]
    ].drop_duplicates("driver_id")
    out = out.drop(
        columns=[
            "quali_position", "grid", "grid_vs_quali", "best_quali_seconds", "q3_seconds",
            "quali_vs_teammate",
        ],
        errors="ignore",
    )
    out = out.merge(q_merge, on="driver_id", how="left")
    out["grid"] = out["quali_position"]
    out["grid_vs_quali"] = 0.0
    teammate_best = out.groupby("constructor_id")["quali_position"].transform("min")
    out["quali_vs_teammate"] = out["quali_position"] - teammate_best

    sp = _raw_sprint_for_round(season, round_num)
    out = out.drop(columns=["sprint_position", "sprint_points"], errors="ignore")
    if not sp.empty:
        out = out.merge(
            sp[["driver_id", "sprint_position", "sprint_points"]],
            on="driver_id",
            how="left",
        )
        out["has_sprint"] = 1
        out["sprint_position"] = out["sprint_position"].fillna(-1)
    else:
        out["has_sprint"] = 0
        out["sprint_position"] = -1
        out["sprint_points"] = np.nan
    return out


def build_upcoming_race_features(season: int, round_num: int) -> pd.DataFrame:
    """Build pre-race feature rows before qualifying/results exist."""
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
    champ_view = champ[["driver_id", "champ_position_before", "champ_points_before", "champ_wins_before"]]
    out = out.drop(columns=["champ_position_before", "champ_points_before", "champ_wins_before"], errors="ignore")
    out = out.merge(champ_view, on="driver_id", how="left")

    for w in (3, 5):
        out[f"driver_id_points_last_{w}"] = (
            prior.groupby("driver_id")["points"].apply(lambda s: s.tail(w).sum()).reindex(out["driver_id"]).values
        )
        out[f"driver_id_avg_finish_last_{w}"] = (
            prior.groupby("driver_id")["position"].apply(lambda s: s.tail(w).mean()).reindex(out["driver_id"]).values
        )
        out[f"driver_id_dnf_rate_last_{w}"] = (
            prior.groupby("driver_id")["dnf"].apply(lambda s: s.tail(w).mean()).reindex(out["driver_id"]).values
        )

    team_hist = (
        prior.groupby(["round", "constructor_id"], as_index=False)
        .agg(team_pts=("points", "sum"), team_best=("position", "min"), team_dnf=("dnf", "mean"))
        .sort_values("round")
    )
    for w in (3, 5):
        out[f"constructor_id_points_last_{w}"] = (
            team_hist.groupby("constructor_id")["team_pts"].apply(lambda s: s.tail(w).sum()).reindex(out["constructor_id"]).values
        )
        out[f"constructor_id_avg_finish_last_{w}"] = (
            team_hist.groupby("constructor_id")["team_best"].apply(lambda s: s.tail(w).mean()).reindex(out["constructor_id"]).values
        )
        out[f"constructor_id_dnf_rate_last_{w}"] = (
            team_hist.groupby("constructor_id")["team_dnf"].apply(lambda s: s.tail(w).mean()).reindex(out["constructor_id"]).values
        )

    circuit_id = meta["circuit_id"]
    circuit_hist = history[history["circuit_id"] == circuit_id]
    out["circuit_avg_finish_prior"] = (
        circuit_hist.groupby("driver_id")["position"].mean().reindex(out["driver_id"]).values
    )
    out["circuit_grid_to_finish_delta"] = (
        circuit_hist.groupby("circuit_id")["grid_finish_delta"].mean().iloc[0]
        if "grid_finish_delta" in circuit_hist.columns and len(circuit_hist) >= 3
        else 0.0
    )
    out["teammate_quali_h2h_rate"] = (
        prior.groupby("driver_id")["beat_teammate_quali"].mean().reindex(out["driver_id"]).values
        if "beat_teammate_quali" in prior.columns
        else np.nan
    )

    out = _apply_upcoming_session_data(out, season, round_num, meta, champ_view)
    out["is_wet"] = 0
    out["precipitation_mm"] = 0.0
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
    out = race_df.copy()
    if is_wet is not None:
        out["is_wet"] = int(is_wet)
        out["precipitation_mm"] = out["precipitation_mm"].fillna(5.0).clip(lower=1.0) if is_wet else 0.0
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


def weekend_card(
    race_df: pd.DataFrame,
    race_mode: RaceMode | None = None,
    kind: ModelKind = "hgb",
) -> pd.DataFrame:
    """
    Full weekend forecast: win, podium, DNF, expected finish, fantasy points.
    Automatically selects pre_quali vs post_quali model.
    """
    mode_ui = race_mode or race_status(race_df)
    wmode = inference_mode(mode_ui)
    bundle = load_weekend_models(wmode)

    if kind == "logreg" and bundle.win_logreg is not None:
        win_model = bundle.win_logreg
    else:
        win_model = bundle.win_model

    out = race_df.copy()
    raw = pd.Series(raw_win_scores(win_model, race_df, wmode), index=race_df.index)
    out["win_prob_raw"] = raw
    out["win_prob"] = apply_race_probs(race_df, raw, temperature=bundle.temperature)
    out["podium_prob"] = predict_binary_proba(bundle.podium_model, race_df, wmode)
    out["dnf_prob"] = predict_binary_proba(bundle.dnf_model, race_df, wmode)
    out["expected_finish"] = bundle.finish_model.predict(feature_matrix(race_df, wmode))
    out["expected_fantasy_pts"] = expected_driver_fantasy_points(out)
    out["model_pick"] = out["win_prob"] == out["win_prob"].max()
    out["inference_mode"] = wmode
    return out.sort_values("win_prob", ascending=False).reset_index(drop=True)


def score_race(
    race_df: pd.DataFrame,
    kind: ModelKind = "hgb",
    normalize: bool = True,
    race_mode: RaceMode | None = None,
) -> pd.DataFrame:
    """Backward-compatible scoring; delegates to weekend_card."""
    scored = weekend_card(race_df, race_mode=race_mode, kind=kind)
    if not normalize:
        scored["win_prob"] = scored["win_prob_raw"]
    return scored


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
        "given_name", "family_name", "constructor_name", "quali_position", "grid",
        "win_prob", "podium_prob", "dnf_prob", "expected_finish", "expected_fantasy_pts",
        "champ_position_before", "is_wet",
    ]
    view = scored[[c for c in cols if c in scored.columns]].copy()
    for c in ("win_prob", "podium_prob", "dnf_prob"):
        if c in view.columns:
            view[c] = (view[c] * 100).round(1)
    if "expected_finish" in view.columns:
        view["expected_finish"] = view["expected_finish"].round(1)
    if "expected_fantasy_pts" in view.columns:
        view["expected_fantasy_pts"] = view["expected_fantasy_pts"].round(1)
    return view.rename(
        columns={
            "given_name": "First",
            "family_name": "Last",
            "constructor_name": "Team",
            "quali_position": "Quali",
            "grid": "Grid",
            "win_prob": "Win %",
            "podium_prob": "Podium %",
            "dnf_prob": "DNF %",
            "expected_finish": "Exp finish",
            "expected_fantasy_pts": "Fantasy pts",
            "champ_position_before": "Champ pos",
            "is_wet": "Wet",
        }
    )


def forecast_tracking(race_df: pd.DataFrame, scored: pd.DataFrame) -> dict:
    """Compare forecast vs actual for completed races."""
    if race_df["won"].max() != 1:
        return {"status": "pending"}
    actual = race_df[race_df["won"] == 1].iloc[0]
    pick = scored.iloc[0]
    top3 = set(scored.head(3)["driver_id"])
    return {
        "status": "complete",
        "actual_winner": f"{actual['given_name']} {actual['family_name']}",
        "model_pick": f"{pick['given_name']} {pick['family_name']}",
        "model_win_prob": float(pick["win_prob"]),
        "correct": actual["driver_id"] == pick["driver_id"],
        "in_top3": actual["driver_id"] in top3,
        "inference_mode": scored["inference_mode"].iloc[0] if "inference_mode" in scored.columns else None,
    }
