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
