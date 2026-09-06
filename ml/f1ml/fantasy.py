"""F1 Fantasy-style expected points engine (approximation of official scoring)."""

from __future__ import annotations

import pandas as pd

# Driver points by race finish (F1 Fantasy 2024-style approximation).
DRIVER_FINISH_POINTS: dict[int, int] = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

QUALI_BONUS: dict[int, int] = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
POSITIONS_GAINED_PER = 1  # +1 per position gained vs grid
POSITIONS_LOST_PER = -1  # -1 per position lost
DNF_PENALTY = -15
FASTEST_LAP_BONUS = 5  # not modeled without FL data; kept for docs


def driver_fantasy_points_row(
    finish: int,
    quali: int | None,
    grid: int | None,
    dnf: bool,
) -> float:
    """Single outcome fantasy points for a driver."""
    if dnf:
        pts = float(DNF_PENALTY)
        if quali is not None and quali in QUALI_BONUS:
            pts += QUALI_BONUS[quali]
        return pts
    pts = float(DRIVER_FINISH_POINTS.get(finish, 0))
    if quali is not None and quali in QUALI_BONUS:
        pts += QUALI_BONUS[quali]
    if grid is not None and finish is not None:
        gained = grid - finish
        if gained > 0:
            pts += gained * POSITIONS_GAINED_PER
        elif gained < 0:
            pts += gained * abs(POSITIONS_LOST_PER)
    return pts


def expected_driver_fantasy_points(scored: pd.DataFrame) -> pd.Series:
    """
    Expected fantasy points from model outputs on a scored race dataframe.

    Requires: win_prob, podium_prob, dnf_prob, expected_finish, quali_position, grid.
    Uses a simplified finish distribution anchored on expected_finish.
    """
    exp_pts = pd.Series(0.0, index=scored.index)
    n = len(scored)
    for idx, row in scored.iterrows():
        ef = float(row.get("expected_finish", n / 2))
        dnf_p = float(row.get("dnf_prob", 0.1))
        quali = row.get("quali_position")
        grid = row.get("grid")
        quali_i = int(quali) if pd.notna(quali) else None
        grid_i = int(grid) if pd.notna(grid) else quali_i

        # Spread probability mass across finish positions around expected finish.
        positions = list(range(1, min(n + 1, 21)))
        weights = []
        for pos in positions:
            dist = abs(pos - ef)
            w = max(0.01, 1.0 / (1.0 + dist))
            weights.append(w)
        total_w = sum(weights)
        finish_probs = [w / total_w * (1.0 - dnf_p) for w in weights]

        pts = 0.0
        for pos, fp in zip(positions, finish_probs):
            pts += fp * driver_fantasy_points_row(pos, quali_i, grid_i, dnf=False)
        pts += dnf_p * driver_fantasy_points_row(0, quali_i, grid_i, dnf=True)
        exp_pts.loc[idx] = pts
    return exp_pts


def constructor_fantasy_points(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expected driver points by constructor."""
    out = scored.copy()
    if "expected_fantasy_pts" not in out.columns:
        out["expected_fantasy_pts"] = expected_driver_fantasy_points(out)
    agg = (
        out.groupby(["constructor_id", "constructor_name"], as_index=False)
        .agg(
            expected_fantasy_pts=("expected_fantasy_pts", "sum"),
            win_prob=("win_prob", "sum"),
            podium_prob=("podium_prob", "sum"),
        )
        .sort_values("expected_fantasy_pts", ascending=False)
    )
    return agg


def oracle_xi(scored: pd.DataFrame, budget: float = 100.0, n_drivers: int = 5) -> pd.DataFrame:
    """
    Greedy budget XI: top expected pts per cost unit.
    Uses placeholder equal cost until official prices are ingested.
    """
    df = scored.copy()
    if "expected_fantasy_pts" not in df.columns:
        df["expected_fantasy_pts"] = expected_driver_fantasy_points(df)
    df["cost"] = 10.0  # placeholder equal pricing
    df["value"] = df["expected_fantasy_pts"] / df["cost"]
    return df.nlargest(n_drivers, "value")[
        ["given_name", "family_name", "constructor_name", "expected_fantasy_pts", "cost", "value"]
    ]
