"""Saved model bundles must load in any supported numpy/scikit-learn environment."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1ml.paths import MODELS
from f1ml.ranking import WeekendRanker, train_dnf_model, train_pairwise_ranker

RNG_TYPES = (np.random.Generator, np.random.BitGenerator, np.random.RandomState)


def _find_rng(obj, seen=None, depth=0):
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 10:
        return None
    seen.add(id(obj))
    if isinstance(obj, RNG_TYPES):
        return type(obj).__name__
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple)):
        children = obj
    elif hasattr(obj, "__dict__"):
        children = vars(obj).values()
    else:
        return None
    for child in children:
        found = _find_rng(child, seen, depth + 1)
        if found:
            return found
    return None


def _training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for rnd in range(1, 9):
        for d in range(10):
            rows.append(
                {
                    "season": 2026,
                    "round": rnd,
                    "driver_id": f"d{d}",
                    "constructor_id": f"t{d // 2}",
                    "position": float(d + 1),
                    "dnf": float(rng.random() < 0.2),
                    "car_pace_ewma": d * 0.2,
                    "quali_position": float(d + 1),
                    "grid": float(d + 1),
                }
            )
    return pd.DataFrame(rows)


def test_saved_bundle_contains_no_numpy_generators(tmp_path: Path):
    # A numpy 2.x Generator pickled into a model cannot be loaded under numpy 1.x
    # ("PCG64 is not a known BitGenerator module"), which broke the dashboard.
    frame = _training_frame()
    ranker = train_pairwise_ranker(frame, "post_quali")
    dnf_model, dnf_features = train_dnf_model(frame, "post_quali")
    WeekendRanker("post_quali", ranker, dnf_model, dnf_features, 1.0, 200).save(tmp_path)
    for name in ("ranker.joblib", "dnf_model.joblib", "meta.joblib"):
        import joblib

        assert _find_rng(joblib.load(tmp_path / name)) is None, name
    reloaded = WeekendRanker.load(tmp_path)
    out = reloaded.predict_frame(frame[frame["round"] == 8])
    assert abs(out["win_prob"].sum() - 1.0) < 1e-6


@pytest.mark.parametrize("mode", ["pre_quali", "post_quali", "post_sprint"])
def test_committed_models_are_portable_and_predict(mode):
    path = MODELS / mode
    if not (path / "meta.joblib").exists():
        pytest.skip(f"no committed {mode} model")
    import joblib

    for name in ("ranker.joblib", "dnf_model.joblib"):
        assert _find_rng(joblib.load(path / name)) is None, f"{mode}/{name} pickles a numpy Generator"
    bundle = WeekendRanker.load(path)
    race = _training_frame()
    race = race[race["round"] == 8]
    out = bundle.predict_frame(race)
    assert out["win_prob"].between(0, 1).all()
    assert abs(out["win_prob"].sum() - 1.0) < 1e-6
