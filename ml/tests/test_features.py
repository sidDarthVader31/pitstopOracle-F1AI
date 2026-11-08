"""Tests for feature engineering helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from f1ml.features import _add_circuit_history_features, _is_classified, _merge_pitstops
from f1ml.specs import feature_columns


def test_merge_pitstops_populates_without_duplicate_columns(tmp_path: Path):
    # Round 2 rows get shifted rolling pit stats from round 1 (round 1 stays NaN by design).
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "round": [1, 1, 2, 2],
            "driver_id": ["d1", "d2", "d1", "d2"],
        }
    )
    pits = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "round": [1, 1, 2, 2],
            "driver_id": ["d1", "d2", "d1", "d2"],
            "stop": [1, 1, 2, 1],
            "duration": ["22.5", "23.0", "21.0", "22.0"],
        }
    )
    pit_path = tmp_path / "pitstops.parquet"
    pits.to_parquet(pit_path, index=False)

    with patch("f1ml.features.RAW", tmp_path):
        out = _merge_pitstops(df)
    assert "pit_stops_avg" in out.columns
    assert "pit_time_avg_s" in out.columns
    assert out.loc[out["round"] == 2, "pit_stops_avg"].notna().all()
    assert out.loc[out["round"] == 1, "pit_stops_avg"].isna().all()
    assert out.loc[out["round"] == 1, "pit_time_avg_s"].isna().all()


def test_lapped_status_is_classified_for_dnf_label():
    assert _is_classified("Finished")
    assert _is_classified("Lapped")
    assert _is_classified("+1 Lap")
    assert not _is_classified("Retired")


def test_dnf_label_excludes_nonstarters_and_disqualifications():
    from f1ml.features import _dnf_labels

    labels = _dnf_labels(pd.Series(["Finished", "Lapped", "Retired", "Did not start", "Disqualified", "Withdrew"]))
    assert labels.iloc[:3].tolist() == [0.0, 0.0, 1.0]
    assert labels.iloc[3:].isna().all()


def test_model_feature_allowlists_exclude_unavailable_current_event_inputs():
    pre, _ = feature_columns("pre_quali")
    post, _ = feature_columns("post_quali")
    sprint, _ = feature_columns("post_sprint")
    forbidden = {
        "race_pace_delta", "is_wet", "precipitation_mm", "has_sprint",
        "fp1_best_lap_delta", "fp2_best_lap_delta", "fp3_best_lap_delta",
        "fp_best_delta", "fp_longrun_delta", "sprint_position",
    }
    assert not forbidden.intersection(pre)
    assert not forbidden.intersection(post)
    assert "sprint_position" in sprint
    assert {"quali_position", "quali_gap_s", "grid"}.issubset(post)
    assert not {"quali_position", "quali_gap_s", "grid"}.intersection(pre)


def test_circuit_overtaking_summary_is_shifted_by_event_not_driver_row():
    from f1ml.features import _add_circuit_history_features

    rows = []
    deltas = [(2, -2), (0, 0), (4, -4), (6, -6)]
    for rnd, pair in enumerate(deltas, start=1):
        for driver_id, delta in zip(("a", "b"), pair):
            rows.append(
                {
                    "season": 2024,
                    "round": rnd,
                    "date": f"2024-{rnd:02d}-01",
                    "circuit_id": "track",
                    "driver_id": driver_id,
                    "constructor_id": driver_id,
                    "grid": 10,
                    "position": 10 - delta,
                }
            )
    out = _add_circuit_history_features(pd.DataFrame(rows))
    for rnd in range(1, 5):
        vals = out.loc[out["round"] == rnd, "circuit_grid_to_finish_delta"]
        assert vals.nunique(dropna=False) == 1
        if rnd <= 3:
            assert vals.isna().all()
    # Race 4 receives the average of the three prior race-level means (0, 0, 0).
    assert out.loc[out["round"] == 4, "circuit_grid_to_finish_delta"].eq(0).all()


def test_grid_importance_uses_only_earlier_races():
    from f1ml.features import add_grid_importance

    rows = []
    # Circuit "lock": finishes exactly in grid order. Circuit "open": reversed.
    for i, (circuit, reverse) in enumerate([("lock", False), ("open", True), ("lock", False), ("open", True)]):
        for g in range(1, 7):
            rows.append(
                {
                    "season": 2024,
                    "round": i + 1,
                    "circuit_id": circuit,
                    "date": f"2024-0{i + 1}-01",
                    "grid": float(g),
                    "position": float(7 - g if reverse else g),
                }
            )
    out = add_grid_importance(pd.DataFrame(rows))
    by_race = out.drop_duplicates("round").set_index("round")["circuit_grid_importance"]
    # Race 1 has no history: neutral prior. Its own result must not leak in.
    assert abs(by_race[1] - 0.6) < 1e-9
    # By round 3, the grid-order circuit scores higher than the reversed one does at round 4.
    assert by_race[3] > by_race[4]


def test_car_class_rating_falls_back_then_tracks_class():
    from f1ml.ratings import MIN_CLASS_RACES, build_ratings

    rows = []
    classes = ["high", "low", "high", "low", "high", "low", "high"]
    for rnd, cls in enumerate(classes, 1):
        # Team "fast_high" is on pole at high-speed tracks and 2% off at low-speed ones.
        rows.append({"season": 2026, "round": rnd, "driver_id": "a", "constructor_id": "fast_high",
                     "best_quali_seconds": 90.0 if cls == "high" else 91.8, "position": 1, "dnf": 0, "speed_class": cls})
        rows.append({"season": 2026, "round": rnd, "driver_id": "b", "constructor_id": "other",
                     "best_quali_seconds": 90.9, "position": 2, "dnf": 0, "speed_class": cls})
    rated = build_ratings(pd.DataFrame(rows))
    team = rated[rated["constructor_id"] == "fast_high"].set_index("round")
    assert team.loc[1, "car_class_delta"] == 0.0  # no class history yet
    last_high = team.loc[7]
    assert MIN_CLASS_RACES <= 3
    assert last_high["car_class_delta"] < 0  # better than its overall rating at high-speed tracks
