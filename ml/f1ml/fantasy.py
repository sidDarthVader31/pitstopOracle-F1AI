"""F1 Fantasy projections, Oracle team and "Check my team" (unofficial helper).

Projections come from ``fantasy_sim.simulate_fantasy`` (the model's own Monte Carlo,
scored with the versioned rules file). Prices come from the hand-entered CSV and are
validated; missing prices block the budget features instead of defaulting.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import pandas as pd

import numpy as np

from f1ml.fantasy_prices import PriceCheck, read_prices, validate_prices
from f1ml.fantasy_rules import ScoringRules, load_rules
from f1ml.fantasy_sim import FantasySim, actual_driver_points, race_seed, simulate_fantasy

# Legacy single-outcome constants, now read from the versioned 2026 rules file
# (data/manual/fantasy_rules_2026.toml). New code uses f1ml.fantasy_sim + ScoringRules.
_RULES_2026 = load_rules(2026)
DRIVER_FINISH_POINTS: dict[int, int] = {i + 1: p for i, p in enumerate(_RULES_2026["race"]["points"])}
QUALI_BONUS: dict[int, int] = {i + 1: p for i, p in enumerate(_RULES_2026["qualifying"]["points"])}
SPRINT_FINISH_POINTS: dict[int, int] = {i + 1: p for i, p in enumerate(_RULES_2026["sprint"]["points"])}
SPRINT_QUALI_BONUS: dict[int, int] = {}  # no sprint-qualifying points in the 2026 rules
POSITIONS_GAINED_PER = _RULES_2026["race"]["position_gained"]
POSITIONS_LOST_PER = _RULES_2026["race"]["position_lost"]
DNF_PENALTY = _RULES_2026["race"]["dnf"]
SPRINT_DNF_PENALTY = _RULES_2026["sprint"]["dnf"]
FASTEST_LAP_BONUS = _RULES_2026["race"]["fastest_lap"]
DEFAULT_BUDGET_M = float(_RULES_2026["team"]["budget_m"])


class PricesMissing(ValueError):
    """Budget features need validated prices for the round; there is no default price."""


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


def _finish_distribution_columns(scored: pd.DataFrame) -> list[tuple[int, str]]:
    cols = []
    for c in scored.columns:
        if c.startswith("finish_p") and c.endswith("_prob"):
            pos = c[len("finish_p"):-len("_prob")]
            if pos.isdigit():
                cols.append((int(pos), c))
    return sorted(cols)


def _approx_finish_probs(ef: float, n: int) -> tuple[list[int], list[float]]:
    """Legacy fallback when no simulated distribution is available.

    Spreads weight 1/(1+|pos-expected|) over every position, which is far flatter
    than the model's real distribution (EXP-007): a clear favourite keeps ~19% on
    finishing outside the points.
    """
    positions = list(range(1, min(n + 1, 21)))
    weights = [max(0.01, 1.0 / (1.0 + abs(pos - ef))) for pos in positions]
    total = sum(weights)
    return positions, [w / total for w in weights]


def expected_driver_fantasy_points(scored: pd.DataFrame, *, use_distribution: bool = True) -> pd.Series:
    """Expected fantasy points from model outputs on a scored race dataframe.

    Uses the Monte Carlo finishing-position distribution (``finish_p{k}_prob``,
    conditional on being classified) times ``1 - dnf_prob`` when it is present, so the
    points reflect the same simulation as the win/podium/points probabilities. Falls
    back to an approximation around ``expected_finish`` otherwise.
    """
    exp_pts = pd.Series(0.0, index=scored.index)
    n = len(scored)
    dist_cols = _finish_distribution_columns(scored) if use_distribution else []
    for idx, row in scored.iterrows():
        ef = float(row.get("expected_finish", n / 2))
        dnf_p = float(row.get("dnf_prob", 0.1))
        quali = row.get("quali_position")
        grid = row.get("grid")
        quali_i = int(quali) if pd.notna(quali) else None
        grid_i = int(grid) if pd.notna(grid) and int(grid) > 0 else quali_i
        has_sprint = int(row.get("has_sprint", 0)) == 1
        sprint_pos = row.get("sprint_position")
        sprint_i = int(sprint_pos) if has_sprint and pd.notna(sprint_pos) and int(sprint_pos) > 0 else None

        dist = [(pos, float(row[c])) for pos, c in dist_cols if pd.notna(row[c])]
        if dist and sum(p for _, p in dist) > 0:
            total = sum(p for _, p in dist)
            positions = [pos for pos, _ in dist]
            cond = [p / total for _, p in dist]
        else:
            positions, cond = _approx_finish_probs(ef, n)
        finish_probs = [p * (1.0 - dnf_p) for p in cond]

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


# ---------------------------------------------------------------------------
# Simulation-based projections (MVP_SPEC §9.3)
# ---------------------------------------------------------------------------


def _is_sprint_weekend(season: int, round_num: int, field: pd.DataFrame | None = None) -> bool:
    from f1ml.schedule import get_weekend

    wk = get_weekend(season, round_num)
    if wk is not None:
        return wk.is_sprint
    return bool(field is not None and "has_sprint" in field and field["has_sprint"].eq(1).any())


def project_weekend(
    scored: pd.DataFrame,
    *,
    season: int,
    round_num: int,
    temperature: float,
    n_samples: int,
    rules: ScoringRules | None = None,
    is_sprint_weekend: bool | None = None,
) -> FantasySim:
    """Fantasy points for every simulated weekend of one race (same seed as the forecast)."""
    rules = rules or load_rules(season)
    sprint = _is_sprint_weekend(season, round_num, scored) if is_sprint_weekend is None else is_sprint_weekend
    return simulate_fantasy(scored, temperature=temperature, n_samples=n_samples, seed=race_seed(season, round_num),
                            rules=rules, is_sprint_weekend=sprint)


def check_prices(season: int, round_num: int, field: pd.DataFrame, rules: ScoringRules | None = None) -> PriceCheck:
    rules = rules or load_rules(season)
    return validate_prices(
        read_prices(season, round_num),
        field["driver_id"].astype(str).drop_duplicates().tolist(),
        field["constructor_id"].astype(str).drop_duplicates().tolist(),
        rules,
    )


def _team_value(d_pts: np.ndarray, c_pts: np.ndarray, boost_multiplier: int) -> float:
    return float(d_pts.sum() + (boost_multiplier - 1) * d_pts.max(initial=0) + c_pts.sum())


def best_team(
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    *,
    budget_m: float,
    n_drivers: int = 5,
    n_constructors: int = 2,
    boost_multiplier: int = 2,
) -> dict | None:
    """Exhaustive search for the highest projected team under budget (2X on its top driver).

    ``drivers``/``constructors`` need id, cost_m, expected. 22C5 x 11C2 ≈ 1.4M teams: vectorised.
    """
    d_ids, d_cost, d_pts = drivers["id"].tolist(), drivers["cost_m"].to_numpy(float), drivers["expected"].to_numpy(float)
    c_ids, c_cost, c_pts = constructors["id"].tolist(), constructors["cost_m"].to_numpy(float), constructors["expected"].to_numpy(float)
    if len(d_ids) < n_drivers or len(c_ids) < n_constructors:
        return None
    dc = np.array(list(itertools.combinations(range(len(d_ids)), n_drivers)))
    cc = np.array(list(itertools.combinations(range(len(c_ids)), n_constructors)))
    d_cost_sum, d_pts_sum = d_cost[dc].sum(1), d_pts[dc].sum(1) + (boost_multiplier - 1) * d_pts[dc].max(1)
    c_cost_sum, c_pts_sum = c_cost[cc].sum(1), c_pts[cc].sum(1)
    cost = d_cost_sum[:, None] + c_cost_sum[None, :]
    value = np.where(cost <= budget_m + 1e-9, d_pts_sum[:, None] + c_pts_sum[None, :], -np.inf)
    i, j = np.unravel_index(np.argmax(value), value.shape)
    if not np.isfinite(value[i, j]):
        return None
    team_d = [d_ids[k] for k in dc[i]]
    boost = team_d[int(np.argmax(d_pts[dc[i]]))]
    return {
        "drivers": team_d,
        "constructors": [c_ids[k] for k in cc[j]],
        "boost": boost,
        "cost_m": round(float(cost[i, j]), 1),
        "expected": float(value[i, j]),
    }


def _entity_tables(proj_drivers: pd.DataFrame, proj_constructors: pd.DataFrame, prices: PriceCheck):
    dp, cp = prices.driver_price(), prices.constructor_price()
    d = proj_drivers.assign(id=proj_drivers["driver_id"], cost_m=proj_drivers["driver_id"].map(dp))
    c = proj_constructors.assign(id=proj_constructors["constructor_id"], cost_m=proj_constructors["constructor_id"].map(cp))
    return d.dropna(subset=["cost_m"]), c.dropna(subset=["cost_m"])


def oracle_team(sim: FantasySim, prices: PriceCheck, rules: ScoringRules) -> dict:
    """Highest-projected valid team (a projection, not advice). Requires validated prices."""
    if not prices.ok:
        raise PricesMissing("; ".join(prices.problems) or "prices missing")
    d, c = _entity_tables(sim.driver_summary(), sim.constructor_summary(), prices)
    team = best_team(d, c, budget_m=float(rules["team"]["budget_m"]), boost_multiplier=int(rules["team"]["boost_multiplier"]))
    if team is None:
        raise PricesMissing("no valid team under budget")
    dist = sim.team_distribution(team["drivers"], team["constructors"], team["boost"], int(rules["team"]["boost_multiplier"]))
    team.update(p10=float(np.percentile(dist, 10)), p90=float(np.percentile(dist, 90)))
    return team


def check_my_team(
    sim: FantasySim,
    prices: PriceCheck,
    rules: ScoringRules,
    my_drivers: list[str],
    my_constructors: list[str],
    *,
    free_transfers: int = 2,
    budget_m: float | None = None,
    max_transfers: int = 3,
) -> dict:
    """Projection for one team plus the best plan using 0..max_transfers swaps.

    Transfers beyond ``free_transfers`` cost ``extra_transfer_penalty`` each. Budget: the team
    after the swaps must cost at most ``budget_m`` (the player's cost cap, default $100M).
    """
    team_rules = rules["team"]
    n_d, n_c = int(team_rules["drivers"]), int(team_rules["constructors"])
    if len(set(my_drivers)) != n_d or len(set(my_constructors)) != n_c:
        raise ValueError(f"pick exactly {n_d} drivers and {n_c} constructors")
    if not prices.ok:
        raise PricesMissing("; ".join(prices.problems) or "prices missing")
    budget = float(budget_m if budget_m is not None else team_rules["budget_m"])
    boost_x = int(team_rules["boost_multiplier"])
    penalty = float(team_rules["extra_transfer_penalty"])
    d, c = _entity_tables(sim.driver_summary(), sim.constructor_summary(), prices)
    d_pts, c_pts = dict(zip(d["id"], d["expected"])), dict(zip(c["id"], c["expected"]))
    d_cost, c_cost = dict(zip(d["id"], d["cost_m"])), dict(zip(c["id"], c["cost_m"]))

    def value(ds, cs):
        return _team_value(np.array([d_pts[x] for x in ds]), np.array([c_pts[x] for x in cs]), boost_x)

    def cost(ds, cs):
        return sum(d_cost[x] for x in ds) + sum(c_cost[x] for x in cs)

    def boost_of(ds):
        return max(ds, key=lambda x: d_pts[x])

    moves = [("driver", o, i) for o in my_drivers for i in d["id"] if i not in my_drivers]
    moves += [("constructor", o, i) for o in my_constructors for i in c["id"] if i not in my_constructors]

    def apply(seq):
        ds, cs = list(my_drivers), list(my_constructors)
        for kind, o, i in seq:
            lst = ds if kind == "driver" else cs
            if o not in lst or i in lst:
                return None
            lst[lst.index(o)] = i
        return ds, cs

    best_by_k: dict[int, tuple[float, tuple]] = {0: (value(my_drivers, my_constructors), ())}
    for k in (1, 2):
        best = None
        for seq in itertools.combinations(moves, k):
            outs = [(m[0], m[1]) for m in seq]
            ins = [(m[0], m[2]) for m in seq]
            if len(set(outs)) < k or len(set(ins)) < k:
                continue
            team = apply(seq)
            if team is None or cost(*team) > budget + 1e-9:
                continue
            v = value(*team)
            if best is None or v > best[0]:
                best = (v, seq)
        if best:
            best_by_k[k] = best
    if max_transfers >= 3 and 2 in best_by_k:  # greedy third swap on top of the best pair
        base = best_by_k[2][1]
        best = None
        for m in moves:
            seq = base + (m,)
            team = apply(seq)
            if team is None or cost(*team) > budget + 1e-9:
                continue
            v = value(*team)
            if best is None or v > best[0]:
                best = (v, seq)
        if best:
            best_by_k[3] = best

    plans = []
    current_value = best_by_k[0][0]
    for k, (v, seq) in sorted(best_by_k.items()):
        team = apply(seq)
        extra = max(0, k - free_transfers)
        dist = sim.team_distribution(team[0], team[1], boost_of(team[0]), boost_x)
        plans.append({
            "n_transfers": k,
            "swaps": [{"kind": kind, "out": o, "in": i,
                       "gain": (d_pts if kind == "driver" else c_pts)[i] - (d_pts if kind == "driver" else c_pts)[o],
                       "cost_change": (d_cost if kind == "driver" else c_cost)[i] - (d_cost if kind == "driver" else c_cost)[o]}
                      for kind, o, i in seq],
            "drivers": team[0], "constructors": team[1], "boost": boost_of(team[0]),
            "expected": v, "p10": float(np.percentile(dist, 10)), "p90": float(np.percentile(dist, 90)),
            "transfer_penalty": extra * penalty,
            "net_gain": v + extra * penalty - current_value,
            "cost_m": round(cost(*team), 1),
        })
    recommended = max(range(len(plans)), key=lambda i: (round(plans[i]["net_gain"], 6), -plans[i]["n_transfers"]))
    return {"current": plans[0], "plans": plans, "recommended": recommended, "budget_m": budget,
            "free_transfers": free_transfers}


def snapshot_fantasy(scored: pd.DataFrame, *, season: int, round_num: int, bundle=None, race=None) -> dict:
    """Fantasy block for a ledger snapshot: per-driver and constructor projections, Oracle team."""
    rules = load_rules(season)
    temperature = float(bundle.temperature) if bundle is not None else 1.0
    n_samples = int(bundle.mc_samples) if bundle is not None else 6000
    sim = project_weekend(scored, season=season, round_num=round_num, temperature=temperature,
                          n_samples=n_samples, rules=rules)
    warnings = []
    if rules.is_placeholder:
        warnings.append(f"Scoring rules {rules.version} are a placeholder, not verified for {season}.")
    if not sim.quali_known:
        warnings.append("Qualifying points are projected from a simulated qualifying order (no qualifying model yet).")
    drivers = {}
    for r in sim.driver_summary().to_dict("records"):
        drivers[r["driver_id"]] = {k: r[k] for k in (
            "expected", "p10", "p50", "p90", "q3_prob", "q2_prob", "expected_quali", "quali_p10", "quali_p90",
            "breakdown", "comparable_expected")}
    names = scored.drop_duplicates("constructor_id").set_index("constructor_id")["constructor_name"].to_dict() \
        if "constructor_name" in scored else {}
    constructors = (sim.constructor_summary().assign(constructor_name=lambda d: d["constructor_id"].map(names))
                    .sort_values("expected", ascending=False).to_dict("records"))
    prices = check_prices(season, round_num, scored, rules)
    out = {"rules_version": rules.version, "prices_version": prices.version, "prices_status": prices.status,
           "drivers": drivers, "constructors": constructors, "warnings": warnings}
    if prices.ok:
        out["oracle_team"] = oracle_team(sim, prices, rules)
    else:
        out["warnings"].append("Fantasy prices for this round are not entered or invalid: no Oracle team. "
                               + "; ".join(prices.problems))
    return out


def sim_from_snapshot(payload: dict) -> FantasySim:
    """Re-create the exact simulated weekends behind a ledger snapshot (for team ranges in the app)."""
    ev = payload["event"]
    field = pd.DataFrame(payload["drivers"])
    if "sim_index" in field:
        field = field.sort_values("sim_index")
    rules = load_rules(int(ev["season"]))
    return project_weekend(field, season=int(ev["season"]), round_num=int(ev["round"]),
                           temperature=float(payload["model"]["temperature"]),
                           n_samples=int(payload.get("mc_samples") or 6000), rules=rules,
                           is_sprint_weekend=ev.get("format") == "sprint")


def _last_n_actual_points(table: pd.DataFrame, season: int, round_num: int, rules: ScoringRules, n: int = 3) -> dict:
    """Baseline: each driver's average comparable fantasy points over their last n races."""
    done = table[(table.groupby(["season", "round"])["won"].transform("max") == 1)
                 & ((table["season"] * 100 + table["round"]) < season * 100 + round_num)]
    keys = done[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).tail(n)
    rows = done.merge(keys, on=["season", "round"])
    pts = []
    for (_, _), g in rows.groupby(["season", "round"]):
        pts.append(actual_driver_points(g, rules, bool(g.get("has_sprint", pd.Series([0])).eq(1).any())))
    if not pts:
        return {}
    return pd.concat(pts).groupby("driver_id")["actual_comparable"].mean().to_dict()


def score_snapshot_fantasy(payload: dict, race: pd.DataFrame, table: pd.DataFrame | None = None) -> dict:
    """Projected vs actual fantasy points for one snapshot (comparable categories only)."""
    ev = payload["event"]
    rules = load_rules(int(ev["season"]))
    actual = actual_driver_points(race, rules, ev.get("format") == "sprint").set_index("driver_id")["actual_comparable"]
    proj = {d["driver_id"]: d.get("fantasy") for d in payload["drivers"] if d.get("fantasy")}
    if not proj:
        return {}
    ids = [d for d in proj if d in actual.index]
    p = np.array([proj[d]["comparable_expected"] for d in ids])
    a = actual.loc[ids].to_numpy(float)
    inside = np.mean([(proj[d]["p10"] <= actual[d] <= proj[d]["p90"]) for d in ids])
    out = {
        "fantasy_mae": round(float(np.abs(p - a).mean()), 2),
        "fantasy_rank_corr": round(float(pd.Series(p).corr(pd.Series(a), method="spearman")), 3),
        "fantasy_range_coverage": round(float(inside), 3),
        "fantasy_scoring_note": "comparable categories: qualifying, finish, places gained/lost, retirement, sprint",
    }
    if table is not None:
        base = _last_n_actual_points(table, int(ev["season"]), int(ev["round"]), rules)
        b_ids = [d for d in ids if d in base]
        if b_ids:
            b = np.array([base[d] for d in b_ids])
            out["baseline_last3_mae"] = round(float(np.abs(b - actual.loc[b_ids].to_numpy(float)).mean()), 2)
    team = payload.get("oracle_team")
    if team:
        boost_x = int(rules["team"]["boost_multiplier"])
        drivers_actual = sum(actual.get(d, 0.0) for d in team["drivers"]) + (boost_x - 1) * actual.get(team["boost"], 0.0)
        out["oracle_team_actual_driver_points"] = round(float(drivers_actual), 1)
        out["oracle_team_projected_driver_points"] = round(float(
            sum(proj[d]["comparable_expected"] for d in team["drivers"] if d in proj)
            + (boost_x - 1) * proj.get(team["boost"], {}).get("comparable_expected", 0.0)), 1)
    return out
