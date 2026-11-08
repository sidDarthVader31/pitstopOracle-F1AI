"""Tests for upcoming-race feature overlay and weekend inference."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from f1ml import predict
from f1ml.features import _merge_fp_pace


@pytest.fixture(autouse=True)
def _clear_predict_caches():
    predict.clear_prediction_caches()
    yield
    predict.clear_prediction_caches()


def test_merge_fp_pace_fp1_fallback(tmp_path: Path):
    fp_path = tmp_path / "fp_pace.parquet"
    pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": [14, 14],
            "driver_id": ["max_verstappen", "lando_norris"],
            "session_type": ["FP1", "FP1"],
            "best_lap_seconds": [90.0, 90.5],
        }
    ).to_parquet(fp_path, index=False)

    df = pd.DataFrame(
        {
            "season": [2026],
            "round": [14],
            "driver_id": ["max_verstappen"],
        }
    )
    with patch("f1ml.features.CANONICAL", tmp_path):
        out = _merge_fp_pace(df)
    assert out.loc[0, "fp1_best_lap_delta"] == pytest.approx(-0.25)
    assert out.loc[0, "fp_best_delta"] == pytest.approx(-0.25)


def test_resolve_race_features_uses_quali_overlay(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    canonical = tmp_path / "canonical"
    raw.mkdir()
    processed.mkdir()
    canonical.mkdir()

    races = pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": [13, 14],
            "race_name": ["Monza", "Baku"],
            "date": ["2026-09-07", "2026-09-14"],
            "circuit_id": ["monza", "baku"],
            "circuit_name": ["Monza", "Baku"],
            "country": ["Italy", "Azerbaijan"],
            "lat": [45.6, 40.4],
            "lng": [9.3, 49.9],
        }
    )
    races.to_parquet(raw / "races.parquet", index=False)

    results = pd.DataFrame(
        {
            "season": [2026] * 2,
            "round": [13] * 2,
            "driver_id": ["max_verstappen", "lando_norris"],
            "constructor_id": ["red_bull", "mclaren"],
            "given_name": ["Max", "Lando"],
            "family_name": ["Verstappen", "Norris"],
            "constructor_name": ["Red Bull", "McLaren"],
            "position": [1, 2],
            "points": [25.0, 18.0],
            "grid": [1, 2],
            "status": ["Finished", "Finished"],
        }
    )
    results.to_parquet(raw / "results.parquet", index=False)

    quali = pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": [13, 14],
            "driver_id": ["max_verstappen", "max_verstappen"],
            "constructor_id": ["red_bull", "red_bull"],
            "given_name": ["Max", "Max"],
            "family_name": ["Verstappen", "Verstappen"],
            "constructor_name": ["Red Bull", "Red Bull"],
            "quali_position": [1, 3],
            "best_quali_seconds": [80.0, 81.0],
            "q3_seconds": [80.0, 81.0],
        }
    )
    quali.to_parquet(raw / "qualifying.parquet", index=False)

    standings = pd.DataFrame(
        {
            "season": [2026, 2026],
            "after_round": [13, 13],
            "driver_id": ["max_verstappen", "lando_norris"],
            "constructor_id": ["red_bull", "mclaren"],
            "standing_position": [1, 2],
            "points": [300.0, 250.0],
            "wins": [10, 5],
        }
    )
    standings.to_parquet(raw / "standings.parquet", index=False)

    pd.DataFrame(
        columns=["season", "round", "precipitation_mm", "rain_mm", "weathercode", "is_wet"]
    ).to_parquet(raw / "weather.parquet", index=False)
    pd.DataFrame(columns=["season", "round", "driver_id", "sprint_position", "sprint_points"]).to_parquet(
        raw / "sprints.parquet", index=False
    )

    history = pd.DataFrame(
        {
            "season": [2026] * 2,
            "round": [13] * 2,
            "driver_id": ["max_verstappen", "lando_norris"],
            "constructor_id": ["red_bull", "mclaren"],
            "given_name": ["Max", "Lando"],
            "family_name": ["Verstappen", "Norris"],
            "constructor_name": ["Red Bull", "McLaren"],
            "race_name": ["Monza", "Monza"],
            "date": ["2026-09-07", "2026-09-07"],
            "circuit_id": ["monza", "monza"],
            "circuit_name": ["Monza", "Monza"],
            "country": ["Italy", "Italy"],
            "position": [1, 2],
            "points": [25.0, 18.0],
            "won": [1, 0],
            "podium": [1, 1],
            "dnf": [0, 0],
            "quali_position": [1, 2],
            "best_quali_seconds": [80.0, 80.5],
            "champ_position_before": [1, 2],
            "champ_points_before": [275.0, 232.0],
            "champ_wins_before": [9, 5],
            "beat_teammate_quali": [0.0, 0.0],
            "grid_finish_delta": [0.0, 0.0],
            "is_wet": [0, 0],
            "precipitation_mm": [8.0, 8.0],
            "race_pace_delta": [4.0, 5.0],
            "fp1_best_lap_delta": [0.2, -0.2],
        }
    )
    history.to_parquet(processed / "driver_race.parquet", index=False)

    with (
        patch("f1ml.predict.RAW", raw),
        patch("f1ml.predict.PROCESSED", processed),
        patch("f1ml.predict.CANONICAL", canonical),
    ):
        predict.clear_prediction_caches()
        race_df, mode = predict.resolve_race_features(2026, 14)

    assert mode == "post_quali"
    max_row = race_df[race_df["driver_id"] == "max_verstappen"].iloc[0]
    assert max_row["quali_position"] == 3
    assert predict.safe_pole_pick(race_df) is not None
    assert race_df["race_pace_delta"].isna().all()
    assert race_df["fp1_best_lap_delta"].isna().all()
    assert race_df["precipitation_mm"].isna().all()
    assert race_df["is_wet"].isna().all()


def test_clear_prediction_caches_invalidates_driver_race(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    path = processed / "driver_race.parquet"
    pd.DataFrame({"season": [2026], "round": [1], "won": [0], "quali_position": [np.nan]}).to_parquet(
        path, index=False
    )
    with patch("f1ml.predict.PROCESSED", processed):
        predict.clear_prediction_caches()
        first = predict.load_driver_race()
        pd.DataFrame({"season": [2026], "round": [1], "won": [1], "quali_position": [1.0]}).to_parquet(
            path, index=False
        )
        predict.clear_prediction_caches()
        second = predict.load_driver_race()
    assert first["won"].iloc[0] == 0
    assert second["won"].iloc[0] == 1


def test_post_sprint_falls_back_to_post_quali_bundle(tmp_path, monkeypatch):
    from f1ml import predict

    calls = []
    (tmp_path / "post_quali").mkdir()
    (tmp_path / "post_quali" / "meta.joblib").write_bytes(b"")
    monkeypatch.setattr(predict, "MODELS", tmp_path)
    monkeypatch.setattr(predict.WeekendRanker, "load", classmethod(lambda cls, path: calls.append(path) or "bundle"))
    predict.load_weekend_models.cache_clear()
    assert predict.load_weekend_models("post_sprint") == "bundle"
    assert calls == [tmp_path / "post_quali"]
    predict.load_weekend_models.cache_clear()
