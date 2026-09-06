"""F1 Fantasy-style expected points and team optimizer."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from f1ml.adapters.manual import ManualAdapter
from f1ml.paths import MANUAL

# Driver points by race finish (F1 Fantasy 2026-style).
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
SPRINT_FINISH_POINTS: dict[int, int] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
SPRINT_QUALI_BONUS: dict[int, int] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
POSITIONS_GAINED_PER = 1
POSITIONS_LOST_PER = -1
DNF_PENALTY = -15
SPRINT_DNF_PENALTY = -5
FASTEST_LAP_BONUS = 5
DEFAULT_BUDGET_M = 100.0
DEFAULT_DRIVER_COST_M = 10.0
DEFAULT_CONSTRUCTOR_COST_M = 15.0

OptimizerMode = Literal["balanced", "safe", "aggressive", "value"]


@dataclass
class FantasyLineup:
    drivers: list[dict]
    constructors: list[dict]
    total_cost_m: float
    expected_points: float
    mode: str


def driver_fantasy_points_row(
    finish: int,
    quali: int | None,
    grid: int | None,
    dnf: bool,
    *,
    sprint_finish: int | None = None,
    sprint_quali: int | None = None,
    sprint_dnf: bool = False,
) -> float:
    """Single outcome fantasy points for a driver."""
    if dnf:
        pts = float(DNF_PENALTY)
        if quali is not None and quali in QUALI_BONUS:
            pts += QUALI_BONUS[quali]
        if sprint_dnf:
            pts += float(SPRINT_DNF_PENALTY)
        elif sprint_finish is not None and sprint_finish in SPRINT_FINISH_POINTS:
            pts += float(SPRINT_FINISH_POINTS[sprint_finish])
        if sprint_quali is not None and sprint_quali in SPRINT_QUALI_BONUS:
            pts += float(SPRINT_QUALI_BONUS[sprint_quali])
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
    if sprint_finish is not None and sprint_finish in SPRINT_FINISH_POINTS:
        pts += float(SPRINT_FINISH_POINTS[sprint_finish])
    if sprint_quali is not None and sprint_quali in SPRINT_QUALI_BONUS:
        pts += float(SPRINT_QUALI_BONUS[sprint_quali])
    return pts


def expected_driver_fantasy_points(scored: pd.DataFrame) -> pd.Series:
    """Expected fantasy points from model outputs on a scored race dataframe."""
    exp_pts = pd.Series(0.0, index=scored.index)
    n = len(scored)
    for idx, row in scored.iterrows():
        ef = float(row.get("expected_finish", n / 2))
        dnf_p = float(row.get("dnf_prob", 0.1))
        quali = row.get("quali_position")
        grid = row.get("grid")
        quali_i = int(quali) if pd.notna(quali) else None
        grid_i = int(grid) if pd.notna(grid) else quali_i
        has_sprint = int(row.get("has_sprint", 0)) == 1
        sprint_pos = row.get("sprint_position")
        sprint_i = int(sprint_pos) if has_sprint and pd.notna(sprint_pos) and int(sprint_pos) > 0 else None

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
            pts += fp * driver_fantasy_points_row(
                pos, quali_i, grid_i, dnf=False, sprint_finish=sprint_i
            )
        pts += dnf_p * driver_fantasy_points_row(
            0, quali_i, grid_i, dnf=True, sprint_finish=sprint_i
        )
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


def load_fantasy_prices(season: int, round_num: int) -> pd.DataFrame:
    adapter = ManualAdapter()
    prices = adapter.load_fantasy_prices(season, round_num)
    if prices.empty and (MANUAL / "fantasy_prices.csv").exists():
        prices = pd.read_csv(MANUAL / "fantasy_prices.csv", comment="#")
        prices = prices[(prices["season"] == season) & (prices["round"] == round_num)]
    return prices


def _attach_prices(scored: pd.DataFrame, season: int, round_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    drivers = scored.copy()
    drivers["entity_type"] = "driver"
    drivers["entity_id"] = drivers["driver_id"]
    drivers["name"] = drivers["given_name"] + " " + drivers["family_name"]
    if "expected_fantasy_pts" not in drivers.columns:
        drivers["expected_fantasy_pts"] = expected_driver_fantasy_points(drivers)

    constructors = constructor_fantasy_points(scored)
    constructors["entity_type"] = "constructor"
    constructors["entity_id"] = constructors["constructor_id"]
    constructors["name"] = constructors["constructor_name"]

    prices = load_fantasy_prices(season, round_num)
    if not prices.empty:
        price_map = prices.set_index(["entity_type", "entity_id"])["cost_m"]
        drivers["cost_m"] = drivers.apply(
            lambda r: float(price_map.get(("driver", r["driver_id"]), DEFAULT_DRIVER_COST_M)),
            axis=1,
        )
        constructors["cost_m"] = constructors.apply(
            lambda r: float(price_map.get(("constructor", r["constructor_id"]), DEFAULT_CONSTRUCTOR_COST_M)),
            axis=1,
        )
    else:
        drivers["cost_m"] = DEFAULT_DRIVER_COST_M
        constructors["cost_m"] = DEFAULT_CONSTRUCTOR_COST_M

    return drivers, constructors


def _mode_score(row: pd.Series, mode: OptimizerMode) -> float:
    pts = float(row["expected_fantasy_pts"])
    cost = float(row["cost_m"])
    if cost <= 0:
        return 0.0
    base = pts / cost
    if mode == "safe":
        dnf_p = float(row.get("dnf_prob", 0.1))
        return base * (1.0 - dnf_p)
    if mode == "aggressive":
        win_p = float(row.get("win_prob", 0.05))
        return base * (1.0 + 2.0 * win_p)
    if mode == "value":
        return pts / (cost ** 1.1)
    return base


def optimize_fantasy_team(
    scored: pd.DataFrame,
    season: int,
    round_num: int,
    *,
    budget_m: float = DEFAULT_BUDGET_M,
    n_drivers: int = 5,
    n_constructors: int = 2,
    mode: OptimizerMode = "balanced",
    max_team_drivers: int = 2,
) -> FantasyLineup:
    """Search best 5 drivers + 2 constructors within budget."""
    drivers, constructors = _attach_prices(scored, season, round_num)
    drivers["score"] = drivers.apply(lambda r: _mode_score(r, mode), axis=1)
    constructors["score"] = constructors["expected_fantasy_pts"] / constructors["cost_m"]

    driver_pool = drivers.nlargest(min(14, len(drivers)), "score")
    constructor_pool = constructors.nlargest(min(8, len(constructors)), "score")

    best: FantasyLineup | None = None
    for d_combo in itertools.combinations(driver_pool.itertuples(index=False), n_drivers):
        team_counts: dict[str, int] = {}
        valid = True
        for d in d_combo:
            cid = str(d.constructor_id)
            team_counts[cid] = team_counts.get(cid, 0) + 1
            if team_counts[cid] > max_team_drivers:
                valid = False
                break
        if not valid:
            continue
        d_cost = sum(float(d.cost_m) for d in d_combo)
        d_pts = sum(float(d.expected_fantasy_pts) for d in d_combo)

        for c_combo in itertools.combinations(constructor_pool.itertuples(index=False), n_constructors):
            total_cost = d_cost + sum(float(c.cost_m) for c in c_combo)
            if total_cost > budget_m:
                continue
            total_pts = d_pts + sum(float(c.expected_fantasy_pts) for c in c_combo)
            lineup = FantasyLineup(
                drivers=[
                    {
                        "name": d.name,
                        "team": d.constructor_name,
                        "cost_m": float(d.cost_m),
                        "expected_pts": float(d.expected_fantasy_pts),
                    }
                    for d in d_combo
                ],
                constructors=[
                    {
                        "name": c.name,
                        "cost_m": float(c.cost_m),
                        "expected_pts": float(c.expected_fantasy_pts),
                    }
                    for c in c_combo
                ],
                total_cost_m=total_cost,
                expected_points=total_pts,
                mode=mode,
            )
            if best is None or lineup.expected_points > best.expected_points:
                best = lineup

    if best is None:
        return FantasyLineup(drivers=[], constructors=[], total_cost_m=0.0, expected_points=0.0, mode=mode)
    return best


def oracle_xi(scored: pd.DataFrame, budget: float = 100.0, n_drivers: int = 5) -> pd.DataFrame:
    """Greedy budget XI (legacy helper)."""
    df = scored.copy()
    if "expected_fantasy_pts" not in df.columns:
        df["expected_fantasy_pts"] = expected_driver_fantasy_points(df)
    df["cost"] = DEFAULT_DRIVER_COST_M
    df["value"] = df["expected_fantasy_pts"] / df["cost"]
    return df.nlargest(n_drivers, "value")[
        ["given_name", "family_name", "constructor_name", "expected_fantasy_pts", "cost", "value"]
    ]


def lineup_to_dataframe(lineup: FantasyLineup) -> pd.DataFrame:
    rows = []
    for d in lineup.drivers:
        rows.append({"Role": "Driver", "Name": d["name"], "Team": d.get("team", ""), "Cost ($M)": d["cost_m"], "Exp pts": d["expected_pts"]})
    for c in lineup.constructors:
        rows.append({"Role": "Constructor", "Name": c["name"], "Team": "", "Cost ($M)": c["cost_m"], "Exp pts": c["expected_pts"]})
    return pd.DataFrame(rows)
