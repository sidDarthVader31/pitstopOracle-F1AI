"""Tests for v3 ranking model, ratings, and race probabilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from f1ml.eval import quali_gap_probabilities, race_probs_from_binary, race_softmax
from f1ml.ranking import (
    market_probs_from_scores,
    predict_dnf_probs,
    predict_scores,
    train_dnf_classifier,
    train_dnf_model,
)
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


def test_constructor_ratings_are_shared_within_each_event():
    df = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "round": [1, 1, 2, 2],
            "driver_id": ["a", "b", "a", "b"],
            "constructor_id": ["team", "team", "team", "team"],
            "best_quali_seconds": [90.0, 95.0, 91.0, 92.0],
            "position": [1, 20, 20, 1],
            "dnf": [0, 1, 1, 0],
            "fp_longrun_delta": [0.0, 4.0, 5.0, 0.0],
            "race_pace_delta": [0.0, 4.0, 5.0, 0.0],
            "fp3_best_lap_delta": [0.0, 4.0, 5.0, 0.0],
            "fp2_best_lap_delta": [0.0, 4.0, 5.0, 0.0],
        }
    )
    rated = build_ratings(df)
    for _key, race in rated.groupby(["season", "round"]):
        assert race["car_pace_ewma"].nunique() == 1
        assert race["dnf_hazard_car"].nunique() == 1


def test_dnf_training_receives_both_labels_even_when_position_missing(monkeypatch):
    seen = {}

    class Recorder:
        def fit(self, x, y, **kwargs):
            seen["labels"] = set(y.astype(int))
            seen["rows"] = len(x)
            return self

    monkeypatch.setattr("f1ml.ranking.make_dnf_pipeline", lambda _features: Recorder())
    monkeypatch.setattr("f1ml.ranking.race_sample_weights", lambda df, eval_year=2026: np.ones(len(df)))
    frame = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "position": [1.0, np.nan, 4.0, np.nan],
            "dnf": [0, 1, 0, 1],
            "car_pace_ewma": [0.0, 0.1, 0.2, 0.3],
            "driver_skill_ewma": [0.0, 0.1, 0.2, 0.3],
            "dnf_hazard_car": [0.1] * 4,
            "dnf_hazard_driver": [0.1] * 4,
            "dnf_hazard_combined": [0.2] * 4,
            "constructor_id_dnf_rate_last_5": [0.1] * 4,
            "driver_id_dnf_rate_last_5": [0.1] * 4,
            "circuit_grid_to_finish_delta": [0.0] * 4,
            "round": [1, 1, 2, 2],
        }
    )
    train_dnf_classifier(frame, "pre_quali")
    assert seen["labels"] == {0, 1}
    assert seen["rows"] == 4


def test_empirical_dnf_model_uses_retirements_and_shrinks_to_field():
    frame = pd.DataFrame(
        {
            "season": [2026] * 12,
            "round": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "constructor_id": ["a", "b"] * 6,
            "position": [1.0, np.nan] * 6,
            "dnf": [0, 1] * 6,
        }
    )
    model, features = train_dnf_model(frame, "pre_quali")
    assert features == ["constructor_id"]
    probs = model.predict(pd.DataFrame({"constructor_id": ["a", "b", "new_team"]}))
    # Team b retired every race, team a never; both shrink toward the 50% field rate.
    assert probs[0] < probs[2] < probs[1]
    assert abs(probs[2] - 0.5) < 1e-9
    assert 0.0 < probs[0] and probs[1] < 1.0


def test_stale_saved_model_features_are_imputed_not_read_from_current_event():
    seen = {}

    class OldRanker:
        named_steps = {
            "pre": SimpleNamespace(transformers=[("num", object(), ["car_driver_combo", "race_pace_delta", "is_wet"])])
        }

        def predict(self, x):
            seen["rank_x"] = x.copy()
            return np.zeros(len(x))

    class OldDnf:
        def predict_proba(self, x):
            seen["dnf_x"] = x.copy()
            return np.column_stack([np.full(len(x), 0.5), np.full(len(x), 0.5)])

    frame = pd.DataFrame(
        {
            "car_driver_combo": [1.0],
            "race_pace_delta": [15.0],
            "is_wet": [1],
            "precipitation_mm": [10.0],
        }
    )
    predict_scores(OldRanker(), frame, "pre_quali")
    predict_dnf_probs(OldDnf(), frame, ["driver_id_dnf_rate_last_5", "is_wet"])
    assert pd.isna(seen["rank_x"].loc[0, "race_pace_delta"])
    assert pd.isna(seen["rank_x"].loc[0, "is_wet"])
    assert pd.isna(seen["dnf_x"].loc[0, "is_wet"])


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
    # Entering 2026 round 1, car rating should be the reset prior, not 2025 pace.
    from f1ml.ratings import NEW_ENTRANT_CAR_PRIOR

    assert abs(r2026["car_pace_ewma"] - NEW_ENTRANT_CAR_PRIOR) < 0.01
    assert abs(r2026["car_quali_class_ewma"] - NEW_ENTRANT_CAR_PRIOR) < 0.01


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
