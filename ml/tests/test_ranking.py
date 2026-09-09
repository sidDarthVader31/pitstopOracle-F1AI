"""Tests for v3 ranking model, ratings, and race probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1ml.eval import quali_gap_probabilities, race_probs_from_binary, race_softmax
from f1ml.ranking import market_probs_from_scores
from f1ml.ratings import ERA_RESET_YEAR, build_ratings


def test_backmarker_not_on_softmax_floor():
    raw = np.array([0.18, 0.015] + [0.01] * 18)
    probs = race_probs_from_binary(raw, temperature=1.0)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[1] < 0.04, f"backmarker prob too high: {probs[1]:.3f}"
    assert probs[0] > probs[1] * 3


def test_old_softmax_on_probs_would_flatten():
    raw = np.array([0.18, 0.015] + [0.01] * 18)
    old = race_softmax(raw, temperature=0.3)
    assert old[1] > 0.04


def test_plackett_luce_probs_sum_to_one():
    scores = np.array([2.0, 1.5, 0.5, -0.5, -1.0])
    dnf = np.array([0.05, 0.05, 0.1, 0.15, 0.2])
    markets = market_probs_from_scores(scores, dnf, n_samples=5000, seed=42)
    assert abs(markets["win_prob"].sum() - 1.0) < 0.05
    assert markets["podium_prob"][0] >= markets["win_prob"][0]


def test_ratings_no_leakage_from_current_race():
    df = pd.DataFrame(
        {
            "season": [2026] * 6,
            "round": [1, 1, 2, 2, 3, 3],
            "driver_id": ["a", "b"] * 3,
            "constructor_id": ["c1", "c2"] * 3,
            "best_quali_seconds": [90.0, 91.0, 89.5, 92.0, 88.0, 93.0],
            "position": [1, 5, 1, 8, 1, 12],
            "dnf": [0, 0, 0, 1, 0, 0],
            "fp_longrun_delta": [0.0, 0.5, -0.2, 0.8, -0.5, 1.0],
            "race_pace_delta": [0.0, 0.3, -0.1, 0.6, -0.3, 0.9],
            "fp3_best_lap_delta": [0.0, 0.5, -0.2, 0.8, -0.5, 1.0],
            "fp2_best_lap_delta": [0.0, 0.5, -0.2, 0.8, -0.5, 1.0],
        }
    )
    rated = build_ratings(df)
    r2 = rated[rated["round"] == 2]
    # Round 2 ratings should reflect only round 1, not round 2 quali.
    assert r2["car_pace_ewma"].notna().all()
    r3 = rated[rated["round"] == 3]
    assert (r3["car_pace_ewma"] != r2["car_pace_ewma"].values[0]).any()


def test_era_reset_constructor_prior():
    df = pd.DataFrame(
        {
            "season": [2025, 2026],
            "round": [22, 1],
            "driver_id": ["a", "a"],
            "constructor_id": ["new_team", "new_team"],
            "best_quali_seconds": [90.0, 95.0],
            "position": [3, 15],
            "dnf": [0, 0],
            "fp_longrun_delta": [0.5, 2.0],
            "race_pace_delta": [0.4, 1.8],
            "fp3_best_lap_delta": [0.5, 2.0],
            "fp2_best_lap_delta": [0.5, 2.0],
        }
    )
    rated = build_ratings(df)
    r2026 = rated[rated["season"] == ERA_RESET_YEAR].iloc[0]
    # Entering 2026 round 1, car rating should be reset prior (0), not 2025 pace.
    assert abs(r2026["car_pace_ewma"]) < 0.01


def test_quali_gap_baseline_not_pole_only():
    df = pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "round": [1, 1, 1],
            "driver_id": ["fast", "slow", "mid"],
            "quali_position": [1, 2, 3],
            "quali_gap_s": [0.0, 0.5, 1.0],
        }
    )
    probs = quali_gap_probabilities(df)
    assert probs.loc[0] > probs.loc[1] > probs.loc[2]
    assert abs(probs.sum() - 1.0) < 1e-9
