#!/usr/bin/env python3
"""Refresh data/manual/weekend_schedule.csv (session start times, UTC) from Jolpica.

    python scripts/sync_schedule.py --season 2026

The file is committed and can be edited by hand if a session moves. The lock time used by
the app is sprint-qualifying start on sprint weekends, else qualifying start (MVP_SPEC §2.1;
the sprint rule is still to be confirmed in the game).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f1ml.paths import JOLPICA_BASE  # noqa: E402
from f1ml.schedule import SCHEDULE_PATH  # noqa: E402


def _ts(session: dict | None) -> str | None:
    if not session or not session.get("date"):
        return None
    return f"{session['date']}T{session.get('time', '00:00:00Z')}".replace("Z", "+00:00")


def fetch(season: int) -> pd.DataFrame:
    resp = requests.get(f"{JOLPICA_BASE}/{season}/races.json", params={"limit": 100}, timeout=30)
    resp.raise_for_status()
    rows = []
    for r in resp.json()["MRData"]["RaceTable"]["Races"]:
        sprint = r.get("Sprint") is not None
        rows.append({
            "season": int(r["season"]),
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "format": "sprint" if sprint else "standard",
            "fp1_start_utc": _ts(r.get("FirstPractice")),
            "sprint_quali_start_utc": _ts(r.get("SprintQualifying") or r.get("SprintShootout")),
            "quali_start_utc": _ts(r.get("Qualifying")),
            "race_start_utc": _ts({"date": r["date"], "time": r.get("time", "00:00:00Z")}),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, required=True)
    a = p.parse_args()
    new = fetch(a.season)
    if SCHEDULE_PATH.exists():
        old = pd.read_csv(SCHEDULE_PATH)
        new = pd.concat([old[old["season"] != a.season], new], ignore_index=True)
    new.sort_values(["season", "round"]).to_csv(SCHEDULE_PATH, index=False)
    print(f"Wrote {len(new)} rows to {SCHEDULE_PATH}")


if __name__ == "__main__":
    main()
