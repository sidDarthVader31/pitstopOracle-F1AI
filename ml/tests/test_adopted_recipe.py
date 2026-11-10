"""The adopted recipe (EXP-005 rolling temperature + EXP-006 classified pace training)
is one code path for the live weekend run and the walk-forward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import race_weekend
from f1ml import ranking, train


def _frame(n_races: int = 25, drivers: int = 6) -> pd.DataFrame:
    rows = []
    for rnd in range(1, n_races + 1):
        for d in range(drivers):
            retired = int(d == 2 and rnd % 3 == 0)
            rows.append({
                "season": 2025, "round": rnd, "date": f"2025-{1 + rnd // 4:02d}-{1 + rnd % 4:02d}",
                "driver_id": f"d{d}", "constructor_id": f"t{d // 2}",
                "position": float(drivers if retired else d + 1), "won": int(d == 0), "dnf": retired,
                "car_pace_ewma": float(d), "quali_position": float(d + 1), "grid": float(d + 1),
                "sprint_position": np.nan, "champ_points_before": 0.0, "champ_position_before": d + 1,
            })
    return pd.DataFrame(rows)


class _Recorder:
    def __init__(self):
        self.n_rows = None

    def fit(self, X, y, reg__sample_weight=None):
        self.n_rows = len(X)
        return self


@pytest.mark.parametrize(("mode", "env", "drops"), [
    ("pre_quali", None, True), ("post_quali", None, True), ("post_sprint", None, False),
    ("post_quali", "all", False),
])
def test_pace_model_training_rows(monkeypatch, mode, env, drops):
    rec = _Recorder()
    monkeypatch.setattr(ranking, "make_rank_pipeline", lambda numeric: rec)
    if env:
        monkeypatch.setenv("F1ML_PACE_TRAIN", env)
    else:
        monkeypatch.delenv("F1ML_PACE_TRAIN", raising=False)
    df = _frame()
    ranking.train_pairwise_ranker(df, mode)
    assert rec.n_rows == (len(df) - int(df["dnf"].sum()) if drops else len(df))


def test_stage_bundle_uses_rolling_temperature_only_with_history(monkeypatch):
    df = _frame()
    monkeypatch.setattr(train, "fit_rolling_temperature", lambda h: 2.5 if h is not None else None)
    bundle, info = train.train_stage_bundle(df, "post_quali", 2025, oos_history=pd.DataFrame({"season": [2024], "round": [1]}))
    assert bundle.temperature == 2.5 and info["temperature_source"] == "rolling_oos"
    bundle, info = train.train_stage_bundle(df, "post_quali", 2025, oos_history=None)
    assert info["temperature_source"] == "last_10_races"
    assert info["pace_train"] == "classified"
    assert train.train_stage_bundle(_frame(n_races=5), "pre_quali", 2025)[0] is None


def test_live_weekend_calls_the_shared_recipe_with_earlier_history_only(tmp_path, monkeypatch):
    hist = pd.DataFrame({
        "season": [2026, 2026, 2026], "round": [4, 5, 6], "stage": "post_quali", "driver_id": "a",
        "pace_score": 0.0, "dnf_prob": 0.1, "won": 1,
    })
    path = tmp_path / "oos.parquet"
    hist.to_parquet(path, index=False)
    monkeypatch.setattr(race_weekend, "OOS_HISTORY_PATH", path)
    calls = {}

    def fake(history, mode, season, *, oos_history=None, mc_samples=0):
        calls.update(mode=mode, rounds=sorted(oos_history["round"]), mc=mc_samples)
        return None, {}

    monkeypatch.setattr(race_weekend, "train_stage_bundle", fake)
    race_weekend.train_weekend_bundle(pd.DataFrame(), "post_quali", 2026, 6)
    assert calls == {"mode": "post_quali", "rounds": [4, 5], "mc": race_weekend.PRODUCTION_MC_SAMPLES}


def test_after_appends_live_forecast_to_history(tmp_path, monkeypatch):
    path = tmp_path / "oos.parquet"
    monkeypatch.setattr(race_weekend, "OOS_HISTORY_PATH", path)
    race = pd.DataFrame({"season": 2026, "round": 7, "driver_id": ["a", "b"], "won": [0, 1]})
    fc = pd.DataFrame({"driver_id": ["a", "b"], "pace_score": [1.0, 0.5], "dnf_prob": [0.1, 0.1]})
    assert race_weekend.append_oos_history(race, fc, "post_quali_pre_sprint") == 2
    assert race_weekend.append_oos_history(race, fc, "post_quali_pre_sprint") == 2  # replaces, no dupes
    assert race_weekend.append_oos_history(race, fc, "post_sprint") == 0
    saved = pd.read_parquet(path)
    assert set(saved["stage"]) == {"post_quali"} and saved["won"].sum() == 1


def test_walk_forward_calibrates_each_fold_on_earlier_forecasts_only(monkeypatch):
    df = _frame(n_races=24)
    seen = []

    class _Bundle:
        temperature = 1.0
        dnf_features = ["constructor_id"]

        class dnf_model:  # noqa: N801
            @staticmethod
            def predict(frame):
                return np.full(len(frame), 0.1)

        def predict_frame(self, frame):
            out = frame.copy()
            n = len(out)
            out["pace_score"] = -out["car_pace_ewma"]
            out["dnf_prob"] = 0.1
            for c in ("win_prob", "podium_prob", "points_prob"):
                out[c] = 1.0 / n
            out["expected_finish"] = (n + 1) / 2
            return out

    def fake(history, mode, eval_year, *, oos_history=None, mc_samples=0):
        last_training_race = max(train._race_keys(history))
        if oos_history is not None:  # only races up to the last training race, never the test race
            assert max(train._race_keys(oos_history)) <= last_training_race
        seen.append((mode, 0 if oos_history is None else oos_history.groupby(["season", "round"]).ngroups))
        return _Bundle(), {}

    monkeypatch.setattr(train, "train_stage_bundle", fake)
    monkeypatch.setattr(train, "predict_dnf_probs", lambda m, frame, f: np.full(len(frame), 0.1))
    metrics: dict = {}
    hist = train._walk_forward_eval(df, metrics)
    pre = [n for m, n in seen if m == "pre_quali"]
    assert pre == list(range(len(pre)))  # history grows by exactly one race per fold
    assert len(metrics["walk_forward"]) == len(pre) == 20  # folds with >= 20 training rows
    assert set(hist["stage"]) == {"pre_quali", "post_quali"}
