"""Expected Fantasy points must follow the simulated finishing distribution (EXP-007)."""

from __future__ import annotations

import pandas as pd
import pytest

from f1ml import fantasy


def _two_driver_race() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["fav", "back"],
            "quali_position": [None, None],
            "grid": [None, None],
            "dnf_prob": [0.1, 0.0],
            "expected_finish": [1.0, 2.0],
            # conditional on being classified
            "finish_p1_prob": [1.0, 0.0],
            "finish_p2_prob": [0.0, 1.0],
        }
    )


def test_expected_points_use_simulated_distribution():
    pts = fantasy.expected_driver_fantasy_points(_two_driver_race())
    # 90% classified P1 (25 pts) + 10% DNF (-15 pts)
    assert pts.iloc[0] == pytest.approx(0.9 * 25 + 0.1 * fantasy.DNF_PENALTY)
    assert pts.iloc[1] == pytest.approx(18.0)


def test_legacy_approximation_is_flatter_than_distribution():
    race = _two_driver_race()
    dist = fantasy.expected_driver_fantasy_points(race)
    legacy = fantasy.expected_driver_fantasy_points(race, use_distribution=False)
    assert legacy.iloc[0] < dist.iloc[0]


def test_falls_back_without_distribution_columns():
    race = _two_driver_race().drop(columns=["finish_p1_prob", "finish_p2_prob"])
    pts = fantasy.expected_driver_fantasy_points(race)
    assert pts.notna().all()
