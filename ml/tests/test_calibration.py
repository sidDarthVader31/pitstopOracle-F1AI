"""EXP-005 calibration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from f1ml import calibration as cal
from f1ml.ranking import market_probs_from_scores


def test_pl_win_probs_match_production_simulation():
    scores = np.array([2.0, 1.0, 0.0, -1.0])
    dnf = np.array([0.1, 0.2, 0.0, 0.05])
    ours = cal.pl_win_probs(scores, cal.dnf_draws(dnf, seed=1, draws=20000), 1.3)
    prod = market_probs_from_scores(scores, dnf, temperature=1.3, n_samples=20000, seed=2)["win_prob"]
    assert ours.sum() == pytest.approx(1.0)
    assert np.allclose(ours, prod, atol=0.01)


def test_blend_weight_zero_and_one():
    p = np.array([0.7, 0.2, 0.1])
    g = np.array([0.4, 0.35, 0.25])
    assert np.allclose(cal.blend(p, g, 0.0), p)
    assert np.allclose(cal.blend(p, g, 1.0), g)


def test_fit_leans_on_prior_when_model_is_overconfident_and_wrong():
    rng = np.random.default_rng(0)
    history = []
    for _ in range(30):
        n = 6
        prior = np.full(n, 1.0 / n)
        scores = rng.normal(size=n) * 4.0  # very sharp but random model
        won = np.zeros(n, dtype=int)
        won[rng.integers(n)] = 1
        history.append(cal.RaceInputs(scores, np.zeros((1, n), dtype=bool), prior, won))
    temperature, weight = cal.fit_grid_blend(history)
    p = cal.calibrated_win_probs(history[0], temperature, weight)
    assert p.max() < 0.5  # a random model must not stay confident
    assert cal.fit_grid_blend([]) == (1.0, 0.0)


# ----------------------------------------------------------------------------- EXP-008
def test_sequential_sampler_keeps_win_probabilities():
    """A softer midfield must not change who is likely to win (same seed, same winner draw)."""
    scores = np.array([2.5, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    retired = cal.dnf_draws(np.full(len(scores), 0.1), seed=3, draws=3000)
    flat = cal.sequential_pl_ranks(scores, retired, 1.0, 1.0, seed=11)
    soft = cal.sequential_pl_ranks(scores, retired, 1.0, 3.0, seed=11)
    np.testing.assert_array_equal(flat == 1, soft == 1)
    exact = cal.pl_win_probs(scores, retired, 1.0)
    np.testing.assert_allclose((flat == 1).mean(axis=0), exact, atol=0.03)


def test_softer_midfield_raises_backmarker_points_chance():
    scores = np.linspace(3, -6, 16)
    retired = np.zeros((4000, 16), dtype=bool)
    flat = cal.outcome_probs_from_ranks(cal.sequential_pl_ranks(scores, retired, 1.0, 1.0, seed=1))
    soft = cal.outcome_probs_from_ranks(cal.sequential_pl_ranks(scores, retired, 1.0, 3.0, seed=1))
    assert soft["points_prob"][-1] > flat["points_prob"][-1]
    assert soft["points_prob"][0] <= flat["points_prob"][0] + 1e-9
    # exactly 10 points finishers per draw when nobody retires
    assert flat["points_prob"].sum() == pytest.approx(10.0)
    assert soft["points_prob"].sum() == pytest.approx(10.0)


def test_retired_drivers_get_rank_zero_and_ranks_are_a_permutation():
    scores = np.array([1.0, 0.0, -1.0, -2.0])
    retired = np.array([[False, True, False, False], [False, False, False, False]])
    ranks = cal.sequential_pl_ranks(scores, retired, 1.0, 2.0, seed=0)
    assert ranks[0, 1] == 0
    assert sorted(ranks[0][ranks[0] > 0]) == [1, 2, 3]
    assert sorted(ranks[1]) == [1, 2, 3, 4]


def test_ramp_schedule_starts_at_one_and_caps_at_p10():
    m = cal.position_multipliers(20, 3.0, "ramp")
    assert m[0] == 1.0 and m[9] == pytest.approx(3.0) and m[-1] == pytest.approx(3.0)
    assert np.all(np.diff(m) >= 0)
    step = cal.position_multipliers(5, 2.0, "step")
    assert step.tolist() == [1.0, 2.0, 2.0, 2.0, 2.0]


def test_rest_multiplier_needs_enough_history_and_picks_lowest_loss():
    few = np.ones((len(cal.REST_MULTIPLIERS), cal.MIN_HISTORY_RACES - 1))
    assert cal.fit_rest_multiplier(few) is None
    losses = np.ones((len(cal.REST_MULTIPLIERS), 12))
    losses[3] = 0.5
    assert cal.fit_rest_multiplier(losses) == cal.REST_MULTIPLIERS[3]
