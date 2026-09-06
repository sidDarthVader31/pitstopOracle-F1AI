"""Tests for race-level win probability conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1ml.eval import race_probs_from_binary, race_softmax
from f1ml.modeling import apply_race_probs


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
    raw = pd.Series([0.2, 0.01, 0.15, 0.02], index=df.index)
    probs = apply_race_probs(df, raw, temperature=1.0)
    assert abs(probs.groupby([df["season"], df["round"]]).sum().sum() - 2.0) < 1e-9
    race1 = probs.iloc[:2]
    assert race1.iloc[1] < 0.05
