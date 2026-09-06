"""Starting grid canonical store and merge logic."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from f1ml.adapters.manual import ManualAdapter
from f1ml.adapters.openf1 import OpenF1Adapter
from f1ml.paths import CANONICAL, MANUAL
from f1ml.schema.entities import StartingGrid

STARTING_GRID_COLUMNS = [
    "season",
    "round",
    "driver_id",
    "grid_position",
    "quali_position",
    "source",
    "as_of",
]


def starting_grid_path():
    return CANONICAL / "starting_grid.parquet"


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STARTING_GRID_COLUMNS)


def load_starting_grid() -> pd.DataFrame:
    path = CANONICAL / "starting_grid.parquet"
    if not path.exists():
        return _empty_frame()
    return pd.read_parquet(path)


def save_starting_grid(df: pd.DataFrame) -> None:
    CANONICAL.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CANONICAL / "starting_grid.parquet", index=False)


def _grids_to_frame(grids: list[StartingGrid]) -> pd.DataFrame:
    if not grids:
        return _empty_frame()
    return pd.DataFrame(
        [
            {
                "season": g.season,
                "round": g.round,
                "driver_id": g.driver_id,
                "grid_position": g.grid_position,
                "quali_position": g.quali_position,
                "source": g.source,
                "as_of": g.as_of,
            }
            for g in grids
        ]
    )


def sync_starting_grid(season: int, round_num: int | None = None) -> pd.DataFrame:
    """Fetch OpenF1 + manual overrides and merge into canonical parquet."""
    existing = load_starting_grid()
    if round_num is None:
        # Default: sync upcoming GP and any rounds in manual CSV only.
        from datetime import date

        from f1ml.predict import next_upcoming_race

        try:
            _, upcoming = next_upcoming_race()
            rounds = {int(upcoming)}
        except Exception:
            rounds = set()
        manual_path = MANUAL / "starting_grid.csv"
        if manual_path.exists():
            manual_df = pd.read_csv(manual_path, comment="#")
            manual_df = manual_df[manual_df["season"] == season]
            rounds |= set(manual_df["round"].astype(int).tolist())
        round_list = sorted(rounds) if rounds else []
    else:
        round_list = [round_num]

    all_new = []
    for rnd in round_list:
        openf1 = OpenF1Adapter().pull(season, rnd)
        manual = ManualAdapter().pull(season, rnd)
        all_new.append(_grids_to_frame(openf1.starting_grids))
        all_new.append(_grids_to_frame(manual.starting_grids))
    frames = [f for f in all_new if not f.empty]
    new_rows = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
    if new_rows.empty:
        return existing

    # Manual overrides beat OpenF1 for same driver/event.
    priority = {"manual": 2, "openf1": 1, "unknown": 0}
    new_rows["_prio"] = new_rows["source"].map(lambda s: priority.get(s, 0))
    new_rows = new_rows.sort_values("_prio", ascending=False).drop_duplicates(
        ["season", "round", "driver_id"], keep="first"
    )
    new_rows = new_rows.drop(columns=["_prio"])

    if not existing.empty:
        keep = existing.copy()
        if round_num is not None:
            keep = keep[~((keep["season"] == season) & (keep["round"] == round_num))]
        else:
            keep = keep[keep["season"] != season]
        merged = pd.concat([keep, new_rows], ignore_index=True)
    else:
        merged = new_rows

    merged = merged.drop_duplicates(["season", "round", "driver_id"], keep="last")
    save_starting_grid(merged)
    return merged


def grid_for_round(season: int, round_num: int) -> pd.DataFrame:
    df = load_starting_grid()
    if df.empty:
        return df
    return df[(df["season"] == season) & (df["round"] == round_num)].copy()


def grid_metadata(season: int, round_num: int) -> dict:
    g = grid_for_round(season, round_num)
    if g.empty:
        return {"has_grid": False}
    sources = g["source"].unique().tolist()
    as_of = g["as_of"].dropna()
    return {
        "has_grid": True,
        "source": "+".join(sources),
        "as_of": as_of.iloc[0] if len(as_of) else None,
        "n_drivers": len(g),
    }


def apply_grid_to_frame(out: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    """Merge published starting grid; fallback quali→grid only when no grid exists."""
    g = grid_for_round(season, round_num)
    if g.empty:
        if "quali_position" in out.columns and out["quali_position"].notna().any():
            out["grid"] = out["quali_position"]
            out["grid_vs_quali"] = 0.0
        return out

    grid_map = g.set_index("driver_id")["grid_position"]
    out = out.copy()
    out["grid"] = out["driver_id"].map(grid_map)
    # Drivers on grid not in lineup get appended via quali merge upstream.
    missing_grid = out["grid"].isna()
    if missing_grid.any() and "quali_position" in out.columns:
        out.loc[missing_grid, "grid"] = out.loc[missing_grid, "quali_position"]
    out["grid_vs_quali"] = out["grid"] - out["quali_position"]
    return out


def penalty_callouts(df: pd.DataFrame) -> list[dict]:
    """Drivers with grid != quali (penalties or promotions)."""
    if "grid" not in df.columns or "quali_position" not in df.columns:
        return []
    sub = df.dropna(subset=["grid", "quali_position"]).copy()
    sub["delta"] = sub["grid"] - sub["quali_position"]
    penalized = sub[sub["delta"] != 0].sort_values("delta", ascending=False)
    out = []
    for row in penalized.itertuples(index=False):
        name = f"{row.given_name} {row.family_name}"
        q = int(row.quali_position)
        g = int(row.grid)
        delta = int(row.delta)
        if delta > 0:
            note = f"P{q} quali → P{g} grid (+{delta} places)"
        else:
            note = f"P{q} quali → P{g} grid ({delta} places)"
        out.append({"driver_id": row.driver_id, "driver_name": name, "note": note, "delta": delta})
    return out


def ensure_manual_template() -> None:
    MANUAL.mkdir(parents=True, exist_ok=True)
    grid_csv = MANUAL / "starting_grid.csv"
    if not grid_csv.exists():
        grid_csv.write_text(
            "season,round,driver_id,grid_position,quali_position,source,as_of\n"
            "# Example: 2026,13,antonelli,20,5,manual,2026-09-06T12:00:00Z\n"
        )
    prices_csv = MANUAL / "fantasy_prices.csv"
    if not prices_csv.exists():
        prices_csv.write_text(
            "season,round,entity_type,entity_id,name,cost_m\n"
            "# entity_type: driver or constructor\n"
            "# Example: 2026,13,driver,max_verstappen,Max Verstappen,30.0\n"
        )
