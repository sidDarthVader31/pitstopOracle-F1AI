import json

import pandas as pd
import pytest

from f1ml.lifecycle import persist_forecast_snapshot, promotion_eligibility, train_candidate


def test_forecast_snapshot_is_immutable_and_excludes_outcomes(tmp_path):
    features = pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": [20, 20],
            "driver_id": ["a", "b"],
            "position": [None, None],
            "won": [0, 0],
            "dnf": [0, 0],
            "grid": [1, 2],
            "grid_as_of": ["2026-12-01T10:00:00Z"] * 2,
        }
    )
    predictions = features.copy()
    predictions["win_prob"] = [0.7, 0.3]
    predictions["dnf_prob"] = [0.1, 0.2]
    predictions["expected_finish"] = [1.2, 1.8]
    predictions["finish_p1_prob"] = [0.7, 0.3]

    destination = persist_forecast_snapshot(
        features,
        predictions,
        "post_quali_pre_sprint",
        "candidate-a",
        tmp_path,
    )
    snapshot = pd.read_parquet(destination / "forecast.parquet")
    manifest = json.loads((destination / "manifest.json").read_text())

    assert "win_prob" in snapshot
    assert "dnf_prob" in snapshot
    assert "position" not in snapshot
    assert "won" not in snapshot
    assert manifest["forecast_stage"] == "post_quali_pre_sprint"
    assert manifest["source_timestamps"]["starting_grid_as_of"] == ["2026-12-01T10:00:00Z"]
    next_destination = persist_forecast_snapshot(
        features, predictions, "post_quali_pre_sprint", "candidate-a", tmp_path
    )
    assert next_destination != destination


def test_candidate_version_must_be_filesystem_safe(tmp_path, monkeypatch):
    from f1ml import lifecycle

    monkeypatch.setattr(lifecycle, "MODELS", tmp_path)
    with pytest.raises(ValueError):
        train_candidate("bad/version")


def test_promotion_gate_requires_chronological_evidence_and_better_scores():
    evidence = {
        "evaluation_type": "chronological_out_of_sample",
        "race_count": 40,
        "minimum_races": 30,
        "candidate_win_log_loss": 0.8,
        "champion_win_log_loss": 0.9,
        "candidate_brier": 0.1,
        "champion_brier": 0.102,
        "candidate_calibration_error": 0.04,
        "champion_calibration_error": 0.05,
        "regulation_era_results": {"2022-2025": {}, "2026": {}},
    }
    eligible, reasons = promotion_eligibility(evidence)
    assert eligible
    assert reasons == []

    evidence["candidate_win_log_loss"] = 0.91
    eligible, reasons = promotion_eligibility(evidence)
    assert not eligible
    assert "candidate does not beat champion on win log loss" in reasons
