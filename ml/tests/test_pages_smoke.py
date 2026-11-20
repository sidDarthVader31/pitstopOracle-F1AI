"""Every MVP page renders without an exception, with an empty and with a populated ledger."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home.py", "fantasy_helper.py", "track_record.py", "how_it_works.py", "scenario_lab.py", "race_explorer.py"]

pytest.importorskip("streamlit.testing.v1")
if not (ROOT / "models" / "pre_quali" / "meta.joblib").exists():
    pytest.skip("needs trained models", allow_module_level=True)


def _run(page: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages" / page), default_timeout=120)
    at.run()
    return at


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    from f1ml import ledger

    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_empty_ledger(page, ledger_dir):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_pages_render_with_a_frozen_snapshot(ledger_dir):
    import pandas as pd

    from f1ml import ledger
    from f1ml.fantasy import snapshot_fantasy
    from f1ml.forecast_snapshot import build_snapshot_payload
    from f1ml.predict import load_weekend_models, next_upcoming_race, resolve_race_features
    from f1ml.schedule import get_weekend

    season, round_num = next_upcoming_race()
    race, mode = resolve_race_features(season, round_num)
    bundle = load_weekend_models("pre_quali")
    bundle.mc_samples = 800
    scored = bundle.predict_frame(race)
    payload = build_snapshot_payload(
        race=race, scored=scored, stage="pre_quali", mode="pre_quali", model_version="weekend-test",
        model_info={"temperature": bundle.temperature, "mc_samples": 800},
        data_as_of=datetime.now(timezone.utc).isoformat(), weekend=get_weekend(season, round_num),
        fantasy=lambda s: snapshot_fantasy(s, season=season, round_num=round_num, bundle=bundle),
    )
    ledger.write_snapshot(payload, ledger_dir)
    for page in ("home.py", "fantasy_helper.py", "track_record.py"):
        at = _run(page)
        assert not at.exception, (page, [e.value for e in at.exception])
