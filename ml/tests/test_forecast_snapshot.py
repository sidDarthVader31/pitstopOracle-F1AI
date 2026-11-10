"""Snapshot payloads follow the output contract; weekly refits are checked before they go live."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from f1ml import ledger
from f1ml.forecast_snapshot import build_snapshot_payload, compute_data_as_of
from f1ml.lifecycle import weekly_refit_checks
from f1ml.schedule import Weekend, countdown_text
from f1ml.specs import feature_columns

LOCK = datetime(2026, 11, 21, 4, 0, tzinfo=timezone.utc)
WEEKEND = Weekend(2026, 21, "Las Vegas Grand Prix", "standard", LOCK, None, datetime(2026, 11, 22, 4, 0, tzinfo=timezone.utc))


def _race_and_scored():
    race = pd.DataFrame({
        "season": [2026, 2026], "round": [21, 21], "driver_id": ["a", "b"],
        "given_name": ["A", "B"], "family_name": ["One", "Two"], "constructor_id": ["x", "y"],
        "constructor_name": ["X", "Y"], "race_name": ["Las Vegas Grand Prix"] * 2, "circuit_name": ["Strip"] * 2,
        "quali_position": [None, None], "grid": [None, None], "won": [0, 0], "date": ["2026-11-22"] * 2,
    })
    scored = race.copy()
    scored["pace_score"] = [1.0, 0.0]
    scored["win_prob"] = [0.7, 0.3]
    scored["podium_prob"] = [1.0, 1.0]
    scored["points_prob"] = [1.0, 1.0]
    scored["dnf_prob"] = [0.1, 0.2]
    scored["expected_finish"] = [1.3, 1.7]
    scored["finish_p1_prob"] = [0.7, 0.3]
    scored["finish_p2_prob"] = [0.3, 0.7]
    return race, scored


def test_payload_has_contract_fields_and_lock_timing(tmp_path):
    race, scored = _race_and_scored()
    common = dict(race=race, scored=scored, stage="pre_quali", mode="pre_quali", model_version="weekend-x",
                  model_info={"races": 100, "temperature": 2.0}, data_as_of="2026-11-08T17:00:00+00:00", weekend=WEEKEND)
    early = build_snapshot_payload(generated_at=datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc), **common)
    late = build_snapshot_payload(generated_at=datetime(2026, 11, 21, 3, 30, tzinfo=timezone.utc), **common)
    assert early["published_before_lock"] is True
    assert late["published_before_lock"] is False  # inside the 1-hour margin
    for key in ("event", "stage", "generated_at", "data_as_of", "model_version", "feature_snapshot_id"):
        assert early[key]
    assert early["drivers"][0]["finish_position_probs"] == [0.7, 0.3]
    assert early["calibration_status"] == "bands_only"
    assert early["feature_snapshot_id"] == late["feature_snapshot_id"]  # same inputs -> same id
    ledger.write_snapshot(early, tmp_path)
    ledger.write_snapshot(late, tmp_path)
    assert ledger.verify_ledger(tmp_path) == []


def test_post_quali_stage_name_is_normalised():
    race, scored = _race_and_scored()
    p = build_snapshot_payload(race=race, scored=scored, stage="post_quali_pre_sprint", mode="post_quali",
                               model_version="v", model_info={}, data_as_of="2026-11-21T04:00:00+00:00", weekend=WEEKEND)
    assert p["stage"] == "post_quali"


def test_data_as_of_uses_latest_finished_race_and_quali():
    table = pd.DataFrame({"season": [2026, 2026, 2026], "round": [19, 20, 21], "won": [1, 1, 0],
                          "date": ["2026-11-01", "2026-11-08", "2026-11-22"]})
    race = pd.DataFrame({"quali_position": [None]})
    assert compute_data_as_of(table, race, WEEKEND).startswith("2026-11-08")
    race = pd.DataFrame({"quali_position": [1.0]})
    assert compute_data_as_of(table, race, WEEKEND).startswith("2026-11-21T04:00")


def _manifest(**overrides):
    modes = {}
    for mode in ("pre_quali", "post_quali"):
        modes[mode] = {"status": "trained", "features": feature_columns(mode)[0], "pace_train": "classified",
                       "temperature": 2.0, "trained_through": "2026-11-08", "races": 110}
        modes[mode].update(overrides)
    return {"modes": modes}


def test_weekly_refit_checks_pass_for_the_approved_recipe():
    assert weekly_refit_checks(_manifest(), race_date="2026-11-22") == []


def test_weekly_refit_checks_refuse_recipe_changes_leakage_and_bad_fits():
    assert any("pace_train" in p for p in weekly_refit_checks(_manifest(pace_train="all")))
    assert any("feature list" in p for p in weekly_refit_checks(_manifest(features=["only_one"])))
    assert any("strictly before" in p for p in weekly_refit_checks(_manifest(trained_through="2026-11-22"), "2026-11-22"))
    assert any("temperature" in p for p in weekly_refit_checks(_manifest(temperature=0.3)))
    assert any("not trained" in p for p in weekly_refit_checks({"modes": {}}))


def test_countdown_text():
    now = datetime(2026, 11, 19, 23, 48, tzinfo=timezone.utc)
    assert countdown_text(LOCK, now) == "1d 04h 12m"
    assert countdown_text(LOCK, LOCK) == "passed"
    assert countdown_text(None) == ""


def test_app_can_reproduce_the_exact_fantasy_draws_from_a_snapshot():
    from f1ml.fantasy import sim_from_snapshot, snapshot_fantasy

    class Bundle:
        temperature, mc_samples = 1.3, 1500

    race, scored = _race_and_scored()
    scored = scored.iloc[::-1].reset_index(drop=True)  # simulation order != win order
    hook = lambda s: snapshot_fantasy(s, season=2026, round_num=21, bundle=Bundle())
    p = build_snapshot_payload(race=race, scored=scored, stage="pre_quali", mode="pre_quali", model_version="v",
                               model_info={"temperature": 1.3, "mc_samples": 1500},
                               data_as_of="2026-11-08T00:00:00+00:00", weekend=WEEKEND, fantasy=hook)
    p["mc_samples"] = 1500
    sim = sim_from_snapshot(p).driver_summary().set_index("driver_id")
    for d in p["drivers"]:
        assert sim.loc[d["driver_id"], "expected"] == d["fantasy"]["expected"]
    assert p["constructors"][0]["expected"] >= p["constructors"][-1]["expected"]
