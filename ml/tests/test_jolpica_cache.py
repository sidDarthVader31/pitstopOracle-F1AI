"""Tests for Jolpica cache policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from f1ml import jolpica


@pytest.fixture(autouse=True)
def _reset_bypass():
    jolpica.clear_bypass_cache_seasons()
    yield
    jolpica.clear_bypass_cache_seasons()


def test_404_responses_are_not_cached(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    with patch.object(jolpica, "CACHE", cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        resp = MagicMock()
        resp.status_code = 404
        with patch.object(jolpica._SESSION, "get", return_value=resp):
            payload = jolpica.get_json("2026/99/pitstops.json")
        assert payload == {"MRData": {}}
        assert list(cache_dir.glob("*.json")) == []


def test_configure_ingest_cache_all_data_clears_range(tmp_path: Path, monkeypatch):
    from f1ml import ingest

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "2024_races.json").write_text("{}")
    (cache_dir / "2025_races.json").write_text("{}")

    monkeypatch.setattr(jolpica, "CACHE", cache_dir)
    with patch("f1ml.ingest.clear_season_caches", wraps=jolpica.clear_season_caches) as cleared:
        ingest.configure_ingest_cache(all_data=True, start_year=2024, end_year=2025)
    cleared.assert_called_once()
    assert not (cache_dir / "2024_races.json").exists()
    assert jolpica._bypass_cache_seasons == {2024, 2025}


def test_configure_ingest_cache_default_bypasses_current_season(monkeypatch):
    from f1ml import ingest

    monkeypatch.setattr("f1ml.ingest.current_season", lambda: 2026)
    with patch("f1ml.ingest.clear_season_caches") as cleared:
        ingest.configure_ingest_cache(all_data=False, start_year=2022, end_year=2026)
    cleared.assert_not_called()
    assert jolpica._bypass_cache_seasons == {2026}
