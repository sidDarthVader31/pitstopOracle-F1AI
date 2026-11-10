"""Append-only, content-hashed forecast ledger (the public track record).

Layout under ``ml/ledger/`` (committed to git, so every published forecast is public):

    snapshots/<season>/R<round>/<stage>__<generated_at>__<sha12>.json   one frozen forecast
    index.jsonl                                                         one line per snapshot
    scores.jsonl                                                        post-race scores of snapshots

Guarantees:

* A snapshot file is created with exclusive-create (``open(..., "x")``): an existing file
  can never be overwritten by this code.
* The file name carries the first 12 hex digits of the SHA-256 of its bytes, and the index
  line carries the full hash, so editing a published file is detectable.
* Each index line includes the hash of the previous index line (a hash chain), so editing,
  deleting or re-ordering earlier index lines is detectable too.
* ``verify_ledger`` re-checks all of the above; ``race_weekend.py`` runs it before writing.

Scores are derived data (snapshot + official result). They are appended, never rewritten;
readers use the latest score line per snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from f1ml.paths import ROOT

LEDGER_ROOT = ROOT / "ledger"
SCHEMA_VERSION = 1
STAGES = ("pre_quali", "post_quali", "post_sprint")
GENESIS = "0" * 64

# Stage names written by older code (lifecycle.persist_forecast_snapshot, reports/forecasts CSVs).
LEGACY_STAGE_NAMES = {"post_quali_pre_sprint": "post_quali"}


class LedgerError(RuntimeError):
    """The ledger is inconsistent or a write would break its guarantees."""


def _clean(value: Any) -> Any:
    """JSON-safe, deterministic values: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def canonical_bytes(payload: dict) -> bytes:
    """Stable serialisation: the same payload always hashes the same."""
    return (json.dumps(_clean(payload), sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_stamp(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: str
    path: Path
    sha256: str
    index_line: dict


def _index_path(root: Path) -> Path:
    return root / "index.jsonl"


def read_index(root: Path | None = None) -> list[dict]:
    path = _index_path(root or LEDGER_ROOT)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _line_hash(entry: dict) -> str:
    return sha256_hex(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def validate_snapshot(payload: dict) -> None:
    """Minimum forecast output contract (ML_MODEL_DESIGN_v3 §7, MVP_SPEC §9.2)."""
    required = ["event", "stage", "generated_at", "data_as_of", "model_version", "feature_snapshot_id", "drivers"]
    missing = [k for k in required if payload.get(k) in (None, "", [])]
    if missing:
        raise LedgerError(f"snapshot is missing required fields: {missing}")
    if payload["stage"] not in STAGES:
        raise LedgerError(f"unknown stage {payload['stage']!r}; expected one of {STAGES}")
    event = payload["event"]
    for k in ("season", "round"):
        if not isinstance(event.get(k), int):
            raise LedgerError(f"event.{k} must be an int")
    ids = [d.get("driver_id") for d in payload["drivers"]]
    if len(ids) != len(set(ids)) or any(not i for i in ids):
        raise LedgerError("drivers must have unique, non-empty driver_id")
    for d in payload["drivers"]:
        for k in ("win_prob", "podium_prob", "points_prob", "dnf_prob"):
            v = d.get(k)
            if v is None or not (0.0 <= float(v) <= 1.0):
                raise LedgerError(f"{d['driver_id']}: {k}={v!r} is not a probability")
    total_win = sum(float(d["win_prob"]) for d in payload["drivers"])
    if abs(total_win - 1.0) > 0.02:
        raise LedgerError(f"win probabilities sum to {total_win:.3f}, expected 1")


def write_snapshot(payload: dict, root: Path | None = None) -> SnapshotRef:
    """Freeze one forecast. Never overwrites; appends one hash-chained index line."""
    root = root or LEDGER_ROOT
    payload = dict(payload)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    validate_snapshot(payload)
    problems = verify_ledger(root)
    if problems:
        raise LedgerError("refusing to append to a ledger that fails verification:\n  " + "\n  ".join(problems))

    season, round_num = payload["event"]["season"], payload["event"]["round"]
    stamp = utc_stamp(datetime.fromisoformat(payload["generated_at"]))
    payload["snapshot_id"] = f"{season}-R{round_num:02d}-{payload['stage']}-{stamp}"
    data = canonical_bytes(payload)
    digest = sha256_hex(data)

    rel = Path("snapshots") / str(season) / f"R{round_num:02d}" / f"{payload['stage']}__{stamp}__{digest[:12]}.json"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as fh:  # exclusive create: fails if the file exists
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError as exc:
        raise LedgerError(f"snapshot already exists, not overwriting: {path}") from exc
    try:
        os.chmod(path, 0o444)  # read-only on disk as a second guard
    except OSError:
        pass

    index = read_index(root)
    if any(e["snapshot_id"] == payload["snapshot_id"] for e in index):
        raise LedgerError(f"snapshot id already in index: {payload['snapshot_id']}")
    entry = {
        "seq": len(index) + 1,
        "snapshot_id": payload["snapshot_id"],
        "season": season,
        "round": round_num,
        "stage": payload["stage"],
        "generated_at": payload["generated_at"],
        "lock_at": payload["event"].get("lock_at"),
        "published_before_lock": payload.get("published_before_lock"),
        "model_version": payload["model_version"],
        "path": rel.as_posix(),
        "sha256": digest,
        "prev": _line_hash(index[-1]) if index else GENESIS,
    }
    with open(_index_path(root), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return SnapshotRef(payload["snapshot_id"], path, digest, entry)


def verify_ledger(root: Path | None = None) -> list[str]:
    """Return every integrity problem found (empty list = ledger is intact)."""
    root = root or LEDGER_ROOT
    problems: list[str] = []
    index = read_index(root)
    prev = GENESIS
    seen_paths: set[str] = set()
    for i, entry in enumerate(index, 1):
        if entry.get("seq") != i:
            problems.append(f"index line {i}: seq is {entry.get('seq')}, expected {i}")
        if entry.get("prev") != prev:
            problems.append(f"index line {i}: hash chain broken (an earlier line was edited, removed or reordered)")
        prev = _line_hash(entry)
        path = root / entry["path"]
        seen_paths.add(entry["path"])
        if not path.exists():
            problems.append(f"{entry['path']}: listed in the index but missing")
            continue
        digest = sha256_hex(path.read_bytes())
        if digest != entry["sha256"]:
            problems.append(f"{entry['path']}: content hash changed since it was published")
        if not path.stem.endswith(entry["sha256"][:12]):
            problems.append(f"{entry['path']}: file name does not match its hash")
    snap_dir = root / "snapshots"
    if snap_dir.exists():
        for path in snap_dir.rglob("*.json"):
            rel = path.relative_to(root).as_posix()
            if rel not in seen_paths:
                problems.append(f"{rel}: snapshot file not in the index")
    return problems


def load_snapshot(entry: dict, root: Path | None = None) -> dict:
    root = root or LEDGER_ROOT
    data = (root / entry["path"]).read_bytes()
    if sha256_hex(data) != entry["sha256"]:
        raise LedgerError(f"{entry['path']}: content hash mismatch")
    return json.loads(data)


def snapshots_for(season: int, round_num: int, root: Path | None = None) -> list[dict]:
    return [e for e in read_index(root) if e["season"] == season and e["round"] == round_num]


def latest_snapshot(season: int, round_num: int, stage: str | None = None, root: Path | None = None) -> dict | None:
    entries = [e for e in snapshots_for(season, round_num, root) if stage is None or e["stage"] == stage]
    return entries[-1] if entries else None


def append_score(score: dict, root: Path | None = None) -> None:
    """Append a post-race score for a snapshot. Scores are never rewritten in place."""
    root = root or LEDGER_ROOT
    if not score.get("snapshot_id"):
        raise LedgerError("a score must reference a snapshot_id")
    if score["snapshot_id"] not in {e["snapshot_id"] for e in read_index(root)}:
        raise LedgerError(f"unknown snapshot_id {score['snapshot_id']}")
    root.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_clean(score), sort_keys=True, separators=(",", ":"))
    with open(root / "scores.jsonl", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_scores(root: Path | None = None) -> dict[str, dict]:
    """Latest score per snapshot_id."""
    path = (root or LEDGER_ROOT) / "scores.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["snapshot_id"]] = row
    return out


def feature_snapshot_id(records: Iterable[dict]) -> str:
    """Hash of the exact model inputs for one race (order-independent by driver)."""
    rows = sorted((_clean(r) for r in records), key=lambda r: str(r.get("driver_id")))
    return "fs-" + sha256_hex(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())[:16]
