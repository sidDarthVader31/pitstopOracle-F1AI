"""Projected F1 Fantasy points from the model's own Monte Carlo (MVP_SPEC §9.2–9.3).

Every simulated race weekend is scored with the versioned rules file, then averaged. So
projected points, likely ranges (10th–90th percentile) and constructor totals all come
from the same draws as the published win/podium chances (same seed, same draws).

What is simulated vs estimated:

* Race finish and retirements: the model's draws (``ranking.simulate_race_draws``).
* Qualifying before it happens: a Plackett–Luce order on the same pace scores, with noise
  correlated with the race draw (``estimates.quali_race_correlation``). There is no
  separate qualifying model yet, so this is a proxy. After qualifying: the real result.
* Starting grid: the official grid when published, else the (simulated or real) qualifying order.
* Sprint (sprint weekends only): simulated the same way, unless the sprint result is known.
* Overtakes: net places gained (lower bound). Fastest lap: one classified driver per
  simulated race, weighted by pace. Driver of the Day: not modelled. Pit stops: a flat rough
  estimate per team. All labelled as estimates in the breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1ml.fantasy_rules import ScoringRules
from f1ml.ranking import simulate_race_draws

DRIVER_COMPONENTS = ("qualifying", "race_finish", "positions", "overtakes", "fastest_lap", "retirement", "sprint")
ESTIMATED_COMPONENTS = ("overtakes", "fastest_lap")
# Categories we can also score from official results, so projected vs actual compares like with like.
COMPARABLE_COMPONENTS = ("qualifying", "race_finish", "positions", "retirement", "sprint")


def race_seed(season: int, round_num: int) -> int:
    """Seed used by ``WeekendRanker.predict_frame`` for this race."""
    return int(season) * 100 + int(round_num)


def _lookup(table: list[int], pos: np.ndarray) -> np.ndarray:
    lut = np.zeros(max(len(table), int(pos.max(initial=0))) + 1)
    lut[1 : len(table) + 1] = table
    return lut[np.clip(pos, 0, len(lut) - 1)]


def _known_positions(values: pd.Series) -> np.ndarray | None:
    """Known positions (1..n) for every driver, or None if unknown. Missing drivers go to the back."""
    v = pd.to_numeric(values, errors="coerce")
    if v.isna().all() or not (v > 0).any():
        return None
    v = v.where(v > 0).to_numpy(dtype=float)
    filled = np.where(np.isnan(v), np.nanmax(v) + 1 + np.arange(len(v)) * 1e-3, v)
    return pd.Series(filled).rank(method="first").to_numpy(dtype=int)


def _order_ranks(perturbed: np.ndarray) -> np.ndarray:
    order = np.argsort(-perturbed, axis=1)
    return np.argsort(order, axis=1) + 1


@dataclass
class FantasySim:
    driver_ids: list[str]
    constructor_of: list[str]
    constructor_ids: list[str]
    components: dict[str, np.ndarray]          # name -> samples x drivers
    total: np.ndarray                          # samples x drivers
    quali: np.ndarray                          # samples x drivers (simulated or known)
    constructor_components: dict[str, np.ndarray]  # name -> samples x constructors
    constructor_total: np.ndarray              # samples x constructors
    rules_version: str
    quali_known: bool
    n_samples: int

    def driver_summary(self) -> pd.DataFrame:
        q3 = self._rules_q3
        rows = []
        for j, d in enumerate(self.driver_ids):
            t = self.total[:, j]
            row = {
                "driver_id": d,
                "constructor_id": self.constructor_of[j],
                "expected": float(t.mean()),
                "p10": float(np.percentile(t, 10)),
                "p50": float(np.percentile(t, 50)),
                "p90": float(np.percentile(t, 90)),
                "q3_prob": float((self.quali[:, j] <= q3).mean()),
                "q2_prob": float((self.quali[:, j] <= self._rules_q2).mean()),
                "expected_quali": float(self.quali[:, j].mean()),
                "quali_p10": int(np.percentile(self.quali[:, j], 10)),
                "quali_p90": int(np.percentile(self.quali[:, j], 90)),
            }
            row["breakdown"] = {k: float(v[:, j].mean()) for k, v in self.components.items()}
            row["comparable_expected"] = float(sum(row["breakdown"][k] for k in COMPARABLE_COMPONENTS))
            rows.append(row)
        return pd.DataFrame(rows)

    def constructor_summary(self) -> pd.DataFrame:
        rows = []
        for k, c in enumerate(self.constructor_ids):
            t = self.constructor_total[:, k]
            rows.append({
                "constructor_id": c,
                "expected": float(t.mean()),
                "p10": float(np.percentile(t, 10)),
                "p50": float(np.percentile(t, 50)),
                "p90": float(np.percentile(t, 90)),
                "breakdown": {name: float(v[:, k].mean()) for name, v in self.constructor_components.items()},
            })
        return pd.DataFrame(rows)

    def team_distribution(self, drivers: list[str], constructors: list[str], boost: str | None = None,
                          boost_multiplier: int = 2) -> np.ndarray:
        """Per-simulation total for one fantasy team (with the 2X Boost driver counted twice)."""
        d_idx = [self.driver_ids.index(d) for d in drivers]
        c_idx = [self.constructor_ids.index(c) for c in constructors]
        total = self.total[:, d_idx].sum(axis=1) + self.constructor_total[:, c_idx].sum(axis=1)
        if boost is not None:
            total = total + (boost_multiplier - 1) * self.total[:, self.driver_ids.index(boost)]
        return total

    _rules_q2: int = 16
    _rules_q3: int = 10


def simulate_fantasy(
    field: pd.DataFrame,
    *,
    temperature: float,
    n_samples: int,
    seed: int,
    rules: ScoringRules,
    is_sprint_weekend: bool = False,
) -> FantasySim:
    """Score every simulated weekend for one race field.

    ``field`` needs driver_id, constructor_id, pace_score, dnf_prob; optional real
    quali_position, grid and sprint_position (used when known).
    """
    field = field.reset_index(drop=True)
    n = len(field)
    scores = field["pace_score"].to_numpy(dtype=float)
    dnf = field["dnf_prob"].to_numpy(dtype=float)
    draws = simulate_race_draws(scores, dnf, temperature, n_samples, seed)
    ranks, retired, adj, g_race = draws["ranks"], draws["retired"], draws["adj"], draws["gumbel"]
    rng = np.random.default_rng(seed + 7919)  # separate stream: never disturbs the race draws
    est = rules["estimates"]
    rho = float(est["quali_race_correlation"])

    def correlated_order(retire_mask: np.ndarray | None = None) -> np.ndarray:
        noise = rho * g_race + np.sqrt(1.0 - rho ** 2) * rng.gumbel(size=(n_samples, n))
        perturbed = adj[None, :] + noise
        if retire_mask is not None:
            perturbed = np.where(retire_mask, -np.inf, perturbed)
        r = _order_ranks(perturbed)
        if retire_mask is not None:
            r = np.where(retire_mask, 0, r)
        return r

    known_q = _known_positions(field["quali_position"]) if "quali_position" in field else None
    quali = np.broadcast_to(known_q, (n_samples, n)).copy() if known_q is not None else correlated_order()
    known_grid = _known_positions(field["grid"]) if "grid" in field else None
    grid = np.broadcast_to(known_grid, (n_samples, n)).copy() if known_grid is not None else quali

    race, q, spr = rules["race"], rules["qualifying"], rules["sprint"]
    classified = ~retired
    comp: dict[str, np.ndarray] = {}
    comp["qualifying"] = _lookup(q["points"], quali)
    comp["race_finish"] = np.where(classified, _lookup(race["points"], ranks), 0.0)
    delta = np.where(classified, grid - ranks, 0)
    comp["positions"] = np.where(delta > 0, delta * race["position_gained"], np.abs(delta) * race["position_lost"])
    comp["overtakes"] = np.clip(delta, 0, None) * race["overtake"]
    fl_pick = np.argmax(np.where(classified, adj[None, :] + rng.gumbel(size=(n_samples, n)), -np.inf), axis=1)
    fl = np.zeros((n_samples, n))
    fl[np.arange(n_samples), fl_pick] = race["fastest_lap"]
    comp["fastest_lap"] = fl
    comp["retirement"] = np.where(retired, float(race["dnf"]), 0.0)

    sprint = np.zeros((n_samples, n))
    if is_sprint_weekend:
        known_s = None
        if "sprint_position" in field:
            sp = pd.to_numeric(field["sprint_position"], errors="coerce")
            if (sp > 0).any():
                known_s = sp
        if known_s is not None:
            s_ranks = np.broadcast_to(np.where(known_s > 0, known_s, 0).astype(int), (n_samples, n))
            s_grid = s_ranks  # sprint grid unknown in our data: no sprint positions-gained term
        else:
            s_ret = rng.random((n_samples, n)) < np.clip(dnf * float(est["sprint_dnf_scale"]), 0, 1)[None, :]
            s_grid = correlated_order()
            s_ranks = correlated_order(s_ret)
        s_class = s_ranks > 0
        s_delta = np.where(s_class, s_grid - s_ranks, 0)
        sprint = (
            np.where(s_class, _lookup(spr["points"], s_ranks), float(spr["dnf"]))
            + np.where(s_delta > 0, s_delta * spr["position_gained"], np.abs(s_delta) * spr["position_lost"])
            + np.clip(s_delta, 0, None) * spr["overtake"]
        )
    comp["sprint"] = sprint
    total = sum(comp.values())

    constructor_of = field["constructor_id"].astype(str).tolist()
    constructor_ids = list(dict.fromkeys(constructor_of))
    con = rules["constructor"]
    c_drivers = np.zeros((n_samples, len(constructor_ids)))
    c_quali = np.zeros_like(c_drivers)
    for k, cid in enumerate(constructor_ids):
        idx = [j for j, c in enumerate(constructor_of) if c == cid]
        c_drivers[:, k] = total[:, idx].sum(axis=1)  # DOTD not modelled, so nothing to exclude
        in_q3 = (quali[:, idx] <= q["q3_cutoff"]).sum(axis=1)
        in_q2 = (quali[:, idx] <= q["q2_cutoff"]).sum(axis=1)
        both = len(idx)
        c_quali[:, k] = np.select(
            [in_q3 >= min(2, both), in_q3 >= 1, in_q2 >= min(2, both), in_q2 >= 1],
            [con["quali_both_q3"], con["quali_one_q3"], con["quali_both_q2"], con["quali_one_q2"]],
            default=con["quali_neither_q2"],
        )
    c_pit = np.full_like(c_drivers, float(est["pit_stop_points_flat"]))
    c_comp = {"drivers": c_drivers, "qualifying_bonus": c_quali, "pit_stops_rough": c_pit}

    sim = FantasySim(
        driver_ids=field["driver_id"].astype(str).tolist(),
        constructor_of=constructor_of,
        constructor_ids=constructor_ids,
        components=comp,
        total=total,
        quali=quali,
        constructor_components=c_comp,
        constructor_total=c_drivers + c_quali + c_pit,
        rules_version=rules.version,
        quali_known=known_q is not None,
        n_samples=n_samples,
    )
    sim._rules_q2, sim._rules_q3 = int(q["q2_cutoff"]), int(q["q3_cutoff"])
    return sim


def actual_driver_points(race: pd.DataFrame, rules: ScoringRules, is_sprint_weekend: bool = False) -> pd.DataFrame:
    """Official-result fantasy points for the comparable categories (no overtakes, FL, DOTD, pit stops)."""
    out = []
    for _, r in race.iterrows():
        pos = pd.to_numeric(r.get("position"), errors="coerce")
        dnf = bool(r.get("dnf") == 1) or pd.isna(pos)
        quali = pd.to_numeric(r.get("quali_position"), errors="coerce")
        grid = pd.to_numeric(r.get("grid"), errors="coerce")
        grid = quali if pd.isna(grid) or grid <= 0 else grid
        q_pts = rules.quali_points(int(quali)) if pd.notna(quali) else 0
        if dnf:
            finish_pts, pos_pts, ret = 0.0, 0.0, float(rules["race"]["dnf"])
        else:
            finish_pts = float(rules.race_points(int(pos)))
            d = (grid - pos) if pd.notna(grid) else 0
            pos_pts = d * rules["race"]["position_gained"] if d > 0 else abs(d) * rules["race"]["position_lost"]
            ret = 0.0
        sprint = 0.0
        sp = pd.to_numeric(r.get("sprint_position"), errors="coerce")
        if is_sprint_weekend and pd.notna(sp):
            sprint = float(rules.sprint_points(int(sp))) if sp > 0 else float(rules["sprint"]["dnf"])
        out.append({
            "driver_id": r["driver_id"],
            "constructor_id": r.get("constructor_id"),
            "actual_comparable": float(q_pts + finish_pts + pos_pts + ret + sprint),
        })
    return pd.DataFrame(out)
