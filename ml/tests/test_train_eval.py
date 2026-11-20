"""Population and chronological split tests for model evaluation."""

from pathlib import Path

import pandas as pd

from f1ml import train


def test_training_frame_keeps_non_position_dnf_rows(tmp_path: Path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    frame = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "round": [1, 1, 1],
            "driver_id": ["winner", "retired_a", "retired_b"],
            "position": [1.0, float("nan"), float("nan")],
            "won": [1, 0, 0],
            "dnf": [0, 1, 1],
        }
    )
    frame.to_parquet(processed / "driver_race.parquet", index=False)
    monkeypatch.setattr(train, "PROCESSED", processed)

    loaded = train._load_training_frame()
    assert len(loaded) == 3
    assert loaded["position"].isna().sum() == 2
    assert set(loaded["dnf"]) == {0, 1}


def test_walk_forward_keeps_full_gp_field_and_only_uses_earlier_races():
    rows = []
    for rnd in (1, 2, 3):
        for driver in ("a", "b", "c"):
            rows.append(
                {
                    "season": 2025,
                    "round": rnd,
                    "driver_id": driver,
                    "won": int(rnd == 1 and driver == "a"),
                    "position": 1 if rnd == 1 and driver == "a" else 2,
                    "dnf": 0,
                }
            )
    frame = pd.DataFrame(rows)
    folds = train._walk_forward_folds(frame, eval_year=2025)

    assert len(folds) == 2
    for fold in folds:
        test_key = (fold["test_season"], fold["test_round"])
        assert fold["test"].groupby(["season", "round"]).ngroups == 1
        assert set(fold["test"]["driver_id"]) == {"a", "b", "c"}
        assert all((s, r) < test_key for s, r in train._race_keys(fold["train"]))


def test_eval_field_keeps_disqualified_and_non_starters(tmp_path: Path, monkeypatch):
    """Regression (ML_LAUNCH_READINESS #7): a driver disqualified after the race was a
    contender at forecast time. Dropping them from the scored field leaks hindsight."""
    processed = tmp_path / "processed"
    processed.mkdir()
    frame = pd.DataFrame(
        {
            "season": [2024] * 4 + [2026] * 2,
            "round": [14] * 4 + [21] * 2,
            "driver_id": ["russell", "hamilton", "piastri", "albon", "a", "b"],
            "status": ["Disqualified", "Finished", "Finished", "Did not start", None, None],
            "position": [20.0, 1.0, 2.0, 19.0, float("nan"), float("nan")],
            "won": [0, 1, 0, 0, 0, 0],
            "dnf": [float("nan"), 0.0, 0.0, float("nan"), float("nan"), float("nan")],
            "is_upcoming": [0, 0, 0, 0, 1, 1],
        }
    )
    frame.to_parquet(processed / "driver_race.parquet", index=False)
    monkeypatch.setattr(train, "PROCESSED", processed)

    training = train._load_training_frame()
    field = train._load_eval_field()
    assert set(training["driver_id"]) == {"hamilton", "piastri"}
    assert set(field["driver_id"]) == {"russell", "hamilton", "piastri", "albon"}


def test_walk_forward_test_fold_uses_full_entry_list():
    rows = []
    for rnd in (1, 2):
        for driver in ("a", "b", "dsq"):
            rows.append(
                {
                    "season": 2025,
                    "round": rnd,
                    "driver_id": driver,
                    "won": int(driver == "a"),
                    "position": 1 if driver == "a" else 2,
                    "dnf": float("nan") if driver == "dsq" else 0.0,
                }
            )
    field = pd.DataFrame(rows)
    labelled = field[field["dnf"].notna()]
    folds = train._walk_forward_folds(labelled, eval_year=2025, test_field=field)

    assert len(folds) == 1
    assert set(folds[0]["test"]["driver_id"]) == {"a", "b", "dsq"}
    assert "dsq" not in set(folds[0]["train"]["driver_id"])
