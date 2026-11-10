"""The forecast ledger is append-only: published snapshots can never be overwritten or silently edited."""
from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from f1ml import ledger


def _payload(generated_at: datetime, stage: str = "pre_quali", win=(0.6, 0.4)) -> dict:
    return {
        "event": {"season": 2026, "round": 21, "race_name": "Las Vegas Grand Prix", "lock_at": "2026-11-21T04:00:00+00:00"},
        "stage": stage,
        "generated_at": generated_at.isoformat(),
        "data_as_of": "2026-11-08T00:00:00+00:00",
        "model_version": "weekend-test",
        "feature_snapshot_id": "fs-abc",
        "published_before_lock": True,
        "drivers": [
            {"driver_id": "a", "win_prob": win[0], "podium_prob": 0.9, "points_prob": 0.95, "dnf_prob": 0.1},
            {"driver_id": "b", "win_prob": win[1], "podium_prob": 0.8, "points_prob": 0.9, "dnf_prob": 0.1},
        ],
    }


T0 = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)


def _bytes_by_path(root) -> dict:
    return {p: p.read_bytes() for p in (root / "snapshots").rglob("*.json")}


def test_existing_snapshots_are_never_overwritten(tmp_path):
    first = ledger.write_snapshot(_payload(T0), tmp_path)
    before = _bytes_by_path(tmp_path)
    index_before = (tmp_path / "index.jsonl").read_text()

    # More snapshots for the same race and stage, including different content.
    ledger.write_snapshot(_payload(T0 + timedelta(hours=1), win=(0.5, 0.5)), tmp_path)
    ledger.write_snapshot(_payload(T0 + timedelta(hours=2), stage="post_quali"), tmp_path)

    after = _bytes_by_path(tmp_path)
    for path, data in before.items():
        assert after[path] == data, f"{path} was modified"
    assert len(after) == 3
    # The index only grows: earlier lines are untouched.
    assert (tmp_path / "index.jsonl").read_text().startswith(index_before)
    assert [e["seq"] for e in ledger.read_index(tmp_path)] == [1, 2, 3]
    assert ledger.verify_ledger(tmp_path) == []
    # Same race, same stage, same second: refused rather than overwritten.
    with pytest.raises(ledger.LedgerError):
        ledger.write_snapshot(_payload(T0), tmp_path)
    assert first.path.read_bytes() == before[first.path]
    assert not os.stat(first.path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_file_name_and_index_carry_the_content_hash(tmp_path):
    ref = ledger.write_snapshot(_payload(T0), tmp_path)
    digest = ledger.sha256_hex(ref.path.read_bytes())
    assert ref.sha256 == digest
    assert ref.path.stem.endswith(digest[:12])
    assert ledger.read_index(tmp_path)[0]["sha256"] == digest
    assert ledger.load_snapshot(ledger.read_index(tmp_path)[0], tmp_path)["snapshot_id"] == ref.snapshot_id


def test_editing_a_published_snapshot_is_detected(tmp_path):
    ref = ledger.write_snapshot(_payload(T0), tmp_path)
    os.chmod(ref.path, 0o644)
    ref.path.write_text(ref.path.read_text().replace("0.6", "0.9"))
    problems = ledger.verify_ledger(tmp_path)
    assert any("content hash changed" in p for p in problems)
    with pytest.raises(ledger.LedgerError):
        ledger.write_snapshot(_payload(T0 + timedelta(hours=1)), tmp_path)


def test_editing_or_deleting_index_lines_breaks_the_chain(tmp_path):
    for h in range(3):
        ledger.write_snapshot(_payload(T0 + timedelta(hours=h)), tmp_path)
    lines = (tmp_path / "index.jsonl").read_text().splitlines()
    edited = json.loads(lines[0])
    edited["model_version"] = "something-else"
    (tmp_path / "index.jsonl").write_text("\n".join([json.dumps(edited, sort_keys=True, separators=(",", ":"))] + lines[1:]) + "\n")
    assert any("hash chain broken" in p for p in ledger.verify_ledger(tmp_path))

    (tmp_path / "index.jsonl").write_text("\n".join(lines[1:]) + "\n")
    assert ledger.verify_ledger(tmp_path)  # removed first line: seq/chain broken + unindexed file


def test_snapshot_must_follow_the_output_contract(tmp_path):
    bad = _payload(T0)
    del bad["feature_snapshot_id"]
    with pytest.raises(ledger.LedgerError, match="feature_snapshot_id"):
        ledger.write_snapshot(bad, tmp_path)
    with pytest.raises(ledger.LedgerError, match="sum"):
        ledger.write_snapshot(_payload(T0, win=(0.9, 0.9)), tmp_path)
    assert not (tmp_path / "index.jsonl").exists()


def test_scores_are_appended_and_latest_wins(tmp_path):
    ref = ledger.write_snapshot(_payload(T0), tmp_path)
    ledger.append_score({"snapshot_id": ref.snapshot_id, "log_loss": 1.0}, tmp_path)
    ledger.append_score({"snapshot_id": ref.snapshot_id, "log_loss": 0.9}, tmp_path)
    assert ledger.read_scores(tmp_path)[ref.snapshot_id]["log_loss"] == 0.9
    assert len((tmp_path / "scores.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ledger.LedgerError):
        ledger.append_score({"snapshot_id": "nope"}, tmp_path)
