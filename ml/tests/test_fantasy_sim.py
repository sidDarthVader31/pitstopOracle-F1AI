"""Fantasy points come from the model's simulated weekends, under a versioned rules file (MVP_SPEC §9.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1ml import fantasy
from f1ml.fantasy_prices import price_template, validate_prices
from f1ml.fantasy_rules import RulesError, load_rules, parse_rules
from f1ml.fantasy_sim import actual_driver_points, simulate_fantasy
from f1ml.ranking import market_probs_from_scores

RULES = load_rules(2026)
TEAMS = ["t1", "t2", "t3", "t4", "t5", "t6"]


def _field(n_per_team=2, quali=None, grid=None) -> pd.DataFrame:
    rows = []
    for k, t in enumerate(TEAMS):
        for j in range(n_per_team):
            i = len(rows)
            rows.append({"driver_id": f"d{i}", "constructor_id": t, "given_name": "D", "family_name": str(i),
                         "constructor_name": t.upper(), "pace_score": 3.0 - 0.25 * i, "dnf_prob": 0.05 + 0.01 * i})
    df = pd.DataFrame(rows)
    df["quali_position"] = quali if quali is not None else np.nan
    df["grid"] = grid if grid is not None else np.nan
    return df


def test_rules_files_are_versioned_and_2027_is_a_placeholder():
    assert RULES.version == "2026.1" and not RULES.is_placeholder
    assert RULES["race"]["dnf"] == -20 and RULES["race"]["fastest_lap"] == 10 and RULES["sprint"]["dnf"] == -10
    r27 = load_rules(2027)
    assert r27.is_placeholder and "placeholder" in r27.version
    assert fantasy.DNF_PENALTY == -20  # legacy constants follow the rules file


def test_rules_validation_rejects_bad_files():
    bad = dict(RULES.raw)
    bad["race"] = dict(bad["race"], points=[1, 25])
    with pytest.raises(RulesError):
        parse_rules(bad)


def test_race_points_match_the_published_finish_distribution():
    field = _field()
    sim = simulate_fantasy(field, temperature=1.2, n_samples=4000, seed=202621, rules=RULES)
    probs = market_probs_from_scores(field["pace_score"].to_numpy(), field["dnf_prob"].to_numpy(), 1.2, 4000, 202621)
    pts = np.array(RULES["race"]["points"] + [0] * (len(field) - 10))
    classified = 1 - probs["dnf_prob"]
    expected_finish_pts = (probs["finish_position_probs"] * pts[None, :]).sum(1) * classified
    assert np.allclose(sim.components["race_finish"].mean(0), expected_finish_pts, atol=1e-9)
    assert np.allclose(sim.components["retirement"].mean(0), -20 * probs["dnf_prob"])


def test_pre_quali_projects_qualifying_points_and_ranges():
    sim = simulate_fantasy(_field(), temperature=1.0, n_samples=3000, seed=1, rules=RULES)
    s = sim.driver_summary()
    assert not sim.quali_known
    assert s.loc[0, "breakdown"]["qualifying"] > 3  # the fastest car scores quali points before quali
    assert (s["p10"] <= s["expected"]).all() and (s["expected"] <= s["p90"]).all()
    assert s.loc[0, "q3_prob"] > s.loc[11, "q3_prob"]
    assert s["expected"].is_monotonic_decreasing or s.loc[0, "expected"] > s.loc[11, "expected"]


def test_known_qualifying_is_deterministic_and_feeds_positions():
    q = list(range(1, 13))
    sim = simulate_fantasy(_field(quali=q, grid=q[::-1]), temperature=1.0, n_samples=2000, seed=3, rules=RULES)
    assert sim.quali_known
    assert sim.components["qualifying"][:, 0].std() == 0 and sim.components["qualifying"][0, 0] == 10
    # d0 qualified P1 but starts P12 (penalty): positions gained must be positive on average
    assert sim.components["positions"][:, 0].mean() > 0


def test_constructor_quali_bonus_levels():
    # t1 both in Q3, t2 one in Q3, t6 neither in Q2 (q2_cutoff 16 / q3_cutoff 10 with 12 cars: use a 22-car field)
    field = _field(n_per_team=2)
    extra = _field(n_per_team=2).assign(driver_id=lambda d: "x" + d["driver_id"], constructor_id=lambda d: d["constructor_id"] + "b")
    field = pd.concat([field, extra.head(10)], ignore_index=True)
    quali = list(range(1, len(field) + 1))
    field["quali_position"] = quali
    sim = simulate_fantasy(field, temperature=1.0, n_samples=200, seed=5, rules=RULES)
    bonus = dict(zip(sim.constructor_ids, sim.constructor_components["qualifying_bonus"][0]))
    assert bonus["t1"] == 10      # P1, P2
    assert bonus["t5"] == 10      # P9, P10
    assert bonus["t6"] == 3       # P11, P12: both Q2
    assert bonus["t4b"] == -1     # P21, P22


def test_prices_are_validated_never_defaulted():
    field = _field()
    tmpl = price_template(2026, 21, field)
    assert tmpl["cost_m"].isna().all()
    ok = tmpl.assign(cost_m=[30.0 - i for i in range(12)] + [25.0, 20.0, 15.0, 10.0, 8.0, 6.0])
    check = validate_prices(ok, field["driver_id"].tolist(), TEAMS, RULES)
    assert check.ok and check.version.startswith("p-")
    bad = pd.concat([ok.iloc[1:], ok.iloc[[2]]]).assign(cost_m=lambda d: d["cost_m"].where(d.index != 3, 55.123))
    problems = " | ".join(validate_prices(bad, field["driver_id"].tolist(), TEAMS, RULES).problems)
    assert "without a price" in problems and "more than once" in problems and "between" in problems
    assert not validate_prices(tmpl.iloc[0:0], field["driver_id"].tolist(), TEAMS, RULES).ok


def _priced(field):
    tmpl = price_template(2026, 21, field)
    costs = {f"d{i}": 30.0 - 2 * i for i in range(12)}
    costs.update({t: 30.0 - 4 * k for k, t in enumerate(TEAMS)})
    tmpl["cost_m"] = tmpl["entity_id"].map(costs)
    return validate_prices(tmpl, field["driver_id"].tolist(), TEAMS, RULES)


def test_oracle_team_needs_prices_and_respects_budget():
    field = _field()
    sim = simulate_fantasy(field, temperature=1.0, n_samples=2000, seed=7, rules=RULES)
    empty = validate_prices(price_template(2026, 21, field).iloc[0:0], field["driver_id"].tolist(), TEAMS, RULES)
    with pytest.raises(fantasy.PricesMissing):
        fantasy.oracle_team(sim, empty, RULES)
    team = fantasy.oracle_team(sim, _priced(field), RULES)
    assert team["cost_m"] <= 100.0 and len(team["drivers"]) == 5 and len(team["constructors"]) == 2
    assert team["boost"] in team["drivers"] and team["p10"] <= team["expected"] <= team["p90"]


def test_check_my_team_charges_for_a_third_transfer():
    field = _field()
    sim = simulate_fantasy(field, temperature=1.0, n_samples=2000, seed=7, rules=RULES)
    prices = _priced(field)
    # A cheap, weak team: lots of room to improve.
    res = fantasy.check_my_team(sim, prices, RULES, ["d7", "d8", "d9", "d10", "d11"], ["t5", "t6"], free_transfers=1)
    by_k = {p["n_transfers"]: p for p in res["plans"]}
    assert by_k[0]["net_gain"] == 0
    assert by_k[2]["transfer_penalty"] == -10
    for k, plan in by_k.items():
        assert plan["transfer_penalty"] == -10 * max(0, k - 1)
        assert plan["net_gain"] == pytest.approx(plan["expected"] - by_k[0]["expected"] + plan["transfer_penalty"])
    for p in res["plans"]:
        assert p["cost_m"] <= 100.0
    rec = res["plans"][res["recommended"]]
    assert rec["net_gain"] == max(p["net_gain"] for p in res["plans"])
    with pytest.raises(ValueError):
        fantasy.check_my_team(sim, prices, RULES, ["d0", "d1"], ["t1", "t2"])


def test_actual_points_use_rules_values():
    race = pd.DataFrame({"driver_id": ["a", "b"], "constructor_id": ["x", "x"], "position": [1, np.nan],
                         "dnf": [0, 1], "quali_position": [3, 1], "grid": [3, 1]})
    pts = actual_driver_points(race, RULES).set_index("driver_id")["actual_comparable"]
    assert pts["a"] == 25 + 8 + 2      # win, P3 quali, +2 places
    assert pts["b"] == 10 - 20         # pole, DNF


def test_check_my_team_treats_current_team_value_as_affordable():
    field = _field()
    sim = simulate_fantasy(field, temperature=1.0, n_samples=1000, seed=7, rules=RULES)
    # The best drivers cost far more than $100M together: still a valid starting point.
    res = fantasy.check_my_team(sim, _priced(field), RULES, ["d0", "d1", "d2", "d3", "d4"], ["t1", "t2"], budget_m=100.0)
    assert res["budget_raised_to_team_value"] and res["budget_m"] == res["current"]["cost_m"]
    assert all(p["cost_m"] <= res["budget_m"] + 1e-9 for p in res["plans"])
