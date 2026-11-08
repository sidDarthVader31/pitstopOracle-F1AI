"""Starting grid (with penalties) must replace qualifying order wherever it is known."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from f1ml.adapters import openf1
from f1ml.features import _apply_upcoming_grid, normalise_grid


def _write_roster(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "round": [20, 20, 20],
            "driver_id": ["hamilton", "leclerc", "rafael_camara"],
            "driver_code": ["HAM", "LEC", None],
            "driver_number": ["44", "16", None],
            "family_name": ["Hamilton", "Leclerc", "Camara"],
            "quali_position": [11, 4, 13],
        }
    ).to_parquet(tmp_path / "qualifying.parquet", index=False)


def test_openf1_grid_comes_from_qualifying_session_and_maps_debutants(tmp_path: Path):
    _write_roster(tmp_path)
    calls = []

    def fake_get_json(path, params=None):
        calls.append((path, params))
        if path == "sessions":
            return [
                {"session_name": "Sprint Qualifying", "session_key": 1},
                {"session_name": "Qualifying", "session_key": 2},
                {"session_name": "Race", "session_key": 3},
            ]
        if path == "starting_grid":
            assert params == {"session_key": 2}, "grid must be read from the Qualifying session"
            return [
                {"position": 14, "driver_number": 16},
                {"position": 20, "driver_number": 44},
                {"position": 11, "driver_number": 51},
            ]
        if path == "drivers":
            return [{"driver_number": 51, "name_acronym": "CAM", "last_name": "Câmara"}]
        return []

    with patch.object(openf1, "RAW", tmp_path), patch.object(openf1, "_get_json", fake_get_json), patch.object(
        openf1, "_meeting_key_for_round", lambda season, rnd: 99
    ):
        grids = openf1.OpenF1Adapter()._fetch_starting_grid(2026, 20)
    by_driver = {g.driver_id: (g.grid_position, g.quali_position) for g in grids}
    assert by_driver["hamilton"] == (20, 11)  # penalty: qualified P11, starts P20
    assert by_driver["leclerc"] == (14, 4)
    assert by_driver["rafael_camara"][0] == 11  # debutant mapped by accent-insensitive name


def test_pit_lane_start_is_treated_as_last_not_unknown():
    df = pd.DataFrame(
        {"season": [2024] * 4, "round": [5] * 4, "driver_id": list("abcd"), "grid": [1.0, 2.0, 0.0, 3.0]}
    )
    assert normalise_grid(df).tolist() == [1.0, 2.0, 4.0, 3.0]


def test_upcoming_race_uses_published_grid_instead_of_qualifying():
    race = pd.DataFrame(
        {
            "season": [2026] * 3,
            "round": [20] * 3,
            "driver_id": ["norris", "russell", "hamilton"],
            "quali_position": [1.0, 2.0, 11.0],
            "grid": [np.nan] * 3,
            "is_upcoming": [1] * 3,
        }
    )
    published = pd.DataFrame(
        {"driver_id": ["norris", "russell", "hamilton"], "grid_position": [1, 7, 20]}
    )
    with patch("f1ml.starting_grid.grid_for_round", lambda season, rnd: published):
        out = _apply_upcoming_grid(race.copy())
    assert out.set_index("driver_id")["grid"].to_dict() == {"norris": 1.0, "russell": 7.0, "hamilton": 20.0}


def test_upcoming_race_falls_back_to_qualifying_when_no_grid_published():
    race = pd.DataFrame(
        {
            "season": [2026] * 2,
            "round": [20] * 2,
            "driver_id": ["norris", "hamilton"],
            "quali_position": [1.0, 11.0],
            "grid": [np.nan] * 2,
            "is_upcoming": [1] * 2,
        }
    )
    with patch("f1ml.starting_grid.grid_for_round", lambda season, rnd: pd.DataFrame()):
        out = _apply_upcoming_grid(race.copy())
    assert out["grid"].tolist() == [1.0, 11.0]


def test_fp_backfill_keeps_existing_round_when_refetch_is_empty(tmp_path: Path):
    from f1ml.adapters import openf1 as of

    pd.DataFrame({"season": [2026], "round": [20], "circuit_id": ["interlagos"]}).to_parquet(
        tmp_path / "races.parquet", index=False
    )
    stored = pd.DataFrame(
        {
            "season": [2026] * 2,
            "round": [20] * 2,
            "driver_id": ["norris", "russell"],
            "session_type": ["FP1", "FP1"],
            "best_lap_seconds": [70.1, 70.3],
        }
    )
    stored.to_parquet(tmp_path / "fp_pace.parquet", index=False)
    with patch.object(of, "RAW", tmp_path), patch.object(of, "CANONICAL", tmp_path), patch.object(
        of.OpenF1Adapter, "_fetch_session_pace", lambda self, season, rnd: []
    ):
        out = of.backfill_fp_pace(2026, 2026, force_rounds={(2026, 20)})
    assert len(out[(out["season"] == 2026) & (out["round"] == 20)]) == 2
