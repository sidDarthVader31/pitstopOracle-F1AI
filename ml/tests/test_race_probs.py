"""Tests for race-level win probability conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1ml.eval import race_probs_from_binary, race_softmax
from f1ml.modeling import apply_race_probs
from f1ml.ranking import market_probs_from_scores, plackett_luce_sample_order


def test_backmarker_not_on_softmax_floor():
    """Independent binary probs must not flatten backmarkers to ~1/N after normalization."""
    raw = np.array([0.18, 0.015] + [0.01] * 18)
    probs = race_probs_from_binary(raw, temperature=1.0)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[1] < 0.04, f"backmarker prob too high: {probs[1]:.3f}"
    assert probs[0] > probs[1] * 3


def test_old_softmax_on_probs_would_flatten():
    """Document the bug: softmax on raw probabilities collapses the field."""
    raw = np.array([0.18, 0.015] + [0.01] * 18)
    old = race_softmax(raw, temperature=0.3)
    assert old[1] > 0.04


def test_apply_race_probs_per_race_group():
    df = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "round": [1, 1, 2, 2],
            "driver_id": ["a", "b", "c", "d"],
        }
    )
    raw = pd.Series([2.0, -1.0, 1.5, -0.5], index=df.index)
    probs = apply_race_probs(df, raw, temperature=1.0)
    assert abs(probs.groupby([df["season"], df["round"]]).sum().sum() - 2.0) < 1e-9
    race1 = probs.iloc[:2]
    assert race1.iloc[0] > race1.iloc[1]
    assert race1.iloc[1] < 0.2


def test_plackett_luce_samples_are_complete_and_reproducible():
    scores = np.array([2.0, 0.5, -1.0])
    a = plackett_luce_sample_order(scores, np.random.default_rng(42), n_samples=200)
    b = plackett_luce_sample_order(scores, np.random.default_rng(42), n_samples=200)
    assert np.array_equal(a, b)
    assert np.all(np.sort(a, axis=1) == np.array([1, 2, 3]))
    assert (a[:, 0] == 1).mean() > (a[:, 2] == 1).mean()


def test_race_simulator_reports_dnf_and_classified_position_distribution():
    result = market_probs_from_scores(
        np.array([2.0, 1.0, 0.0]),
        np.array([0.0, 0.5, 0.99]),
        n_samples=1000,
        seed=17,
    )
    assert np.allclose(result["finish_position_probs"].sum(axis=1), 1.0)
    assert result["dnf_prob"][0] == 0.0
    assert result["dnf_prob"][1] == pytest.approx(0.5, abs=0.05)
    assert result["dnf_prob"][2] > 0.98
    assert result["win_prob"].sum() == pytest.approx(1.0)
    assert result["podium_prob"][2] < 0.03
    assert result["expected_finish"][2] >= 1.0


def test_win_probs_sum_to_one_and_never_exactly_zero():
    import numpy as np

    from f1ml.ranking import market_probs_from_scores

    scores = -np.arange(1, 21, dtype=float) * 2.0  # very confident ordering
    out = market_probs_from_scores(scores, np.full(20, 0.1), temperature=0.3, n_samples=500)
    assert abs(out["win_prob"].sum() - 1.0) < 1e-6
    assert (out["win_prob"] > 0).all()
    assert (out["podium_prob"] >= out["win_prob"] - 1e-12).all()
    assert (out["points_prob"] >= out["podium_prob"] - 1e-12).all()
