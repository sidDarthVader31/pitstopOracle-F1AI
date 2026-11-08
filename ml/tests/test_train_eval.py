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
