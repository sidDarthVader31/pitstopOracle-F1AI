"""Tests for feature engineering helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from f1ml.features import _merge_pitstops


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
