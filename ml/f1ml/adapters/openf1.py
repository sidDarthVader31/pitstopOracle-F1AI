"""OpenF1 adapter — starting grid and practice pace."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from f1ml.adapters.base import SourceAdapter
from f1ml.paths import CACHE, RAW
from f1ml.schema.entities import CanonicalBundle, SessionResult, StartingGrid

OPENF1_BASE = "https://api.openf1.org/v1"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "pitstopOracle-F1AI/1.0"})


def _get_json(path: str, params: dict[str, Any] | None = None) -> list[dict]:
    url = f"{OPENF1_BASE}/{path.lstrip('/')}"
    for attempt in range(4):
        time.sleep(0.25)
        resp = _SESSION.get(url, params=params or {}, timeout=60)
        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    return []


def _driver_number_map(season: int) -> dict[int, str]:
    """Map permanent driver number -> Ergast driver_id from cached Jolpica data."""
    mapping: dict[int, str] = {}
    for name in ("qualifying.parquet", "results.parquet"):
        path = RAW / name
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        sub = df[df["season"] == season][["driver_id", "driver_number"]].dropna()
        for row in sub.drop_duplicates("driver_number").itertuples(index=False):
            try:
                num = int(row.driver_number)
            except (TypeError, ValueError):
                continue
            mapping[num] = str(row.driver_id)
    return mapping


def _race_session_key(season: int, round_num: int) -> int | None:
    meetings = _get_json("meetings", {"year": season})
    if not meetings:
        return None
    races = pd.read_parquet(RAW / "races.parquet")
    race_row = races[(races["season"] == season) & (races["round"] == round_num)]
    if race_row.empty:
        return None
    country = str(race_row.iloc[0].get("country", ""))
    circuit = str(race_row.iloc[0].get("circuit_id", ""))
    meeting_key = None
    for m in meetings:
        if country and m.get("country_name") == country:
            meeting_key = m.get("meeting_key")
            break
        if circuit and circuit.replace("_", " ") in str(m.get("circuit_short_name", "")).lower():
            meeting_key = m.get("meeting_key")
            break
    if meeting_key is None and len(meetings) >= round_num:
        meeting_key = meetings[round_num - 1].get("meeting_key")
    if meeting_key is None:
        return None
    sessions = _get_json("sessions", {"meeting_key": meeting_key})
    for s in sessions:
        if s.get("session_type") == "Race" or s.get("session_name") == "Race":
            return int(s["session_key"])
    return None


class OpenF1Adapter(SourceAdapter):
    name = "openf1"

    def pull(self, season: int, round_num: int | None = None) -> CanonicalBundle:
        bundle = CanonicalBundle()
        if round_num is not None:
            rounds = [round_num]
        else:
            from f1ml.predict import next_upcoming_race

            try:
                s, upcoming = next_upcoming_race()
                rounds = [upcoming] if s == season else []
            except Exception:
                rounds = []
        for rnd in rounds:
            bundle.starting_grids.extend(self._fetch_starting_grid(season, int(rnd)))
            bundle.session_results.extend(self._fetch_fp_pace(season, int(rnd)))
        return bundle

    def _fetch_starting_grid(self, season: int, round_num: int) -> list[StartingGrid]:
        session_key = _race_session_key(season, round_num)
        if session_key is None:
            return []
        rows = _get_json("starting_grid", {"session_key": session_key})
        if not rows:
            return []
        num_map = _driver_number_map(season)
        quali = self._quali_positions(season, round_num)
        as_of = datetime.now(timezone.utc).isoformat()
        out: list[StartingGrid] = []
        for row in rows:
            num = row.get("driver_number")
            if num is None:
                continue
            driver_id = num_map.get(int(num))
            if not driver_id:
                continue
            out.append(
                StartingGrid(
                    season=season,
                    round=round_num,
                    driver_id=driver_id,
                    grid_position=int(row["position"]),
                    quali_position=quali.get(driver_id),
                    source="openf1",
                    as_of=as_of,
                )
            )
        return out

    def _quali_positions(self, season: int, round_num: int) -> dict[str, int]:
        path = RAW / "qualifying.parquet"
        if not path.exists():
            return {}
        q = pd.read_parquet(path)
        sub = q[(q["season"] == season) & (q["round"] == round_num)]
        return {
            str(r.driver_id): int(r.quali_position)
            for r in sub.dropna(subset=["quali_position"]).itertuples(index=False)
        }

    def _fetch_fp_pace(self, season: int, round_num: int) -> list[SessionResult]:
        """Best lap per driver from FP2/FP3 for pace features."""
        meetings = _get_json("meetings", {"year": season})
        if not meetings:
            return []
        races = pd.read_parquet(RAW / "races.parquet")
        race_row = races[(races["season"] == season) & (races["round"] == round_num)]
        if race_row.empty:
            return []
        country = str(race_row.iloc[0].get("country", ""))
        meeting_key = None
        for m in meetings:
            if m.get("country_name") == country:
                meeting_key = m.get("meeting_key")
                break
        if meeting_key is None:
            return []
        sessions = _get_json("sessions", {"meeting_key": meeting_key})
        num_map = _driver_number_map(season)
        results: list[SessionResult] = []
        for sess in sessions:
            stype = sess.get("session_name") or sess.get("session_type") or ""
            if stype not in ("Practice 2", "Practice 3", "FP2", "FP3"):
                continue
            session_key = sess.get("session_key")
            if not session_key:
                continue
            laps = _get_json("laps", {"session_key": session_key})
            if not laps:
                continue
            best_by_driver: dict[str, float] = {}
            for lap in laps:
                dur = lap.get("lap_duration")
                num = lap.get("driver_number")
                if dur is None or num is None:
                    continue
                driver_id = num_map.get(int(num))
                if not driver_id:
                    continue
                dur_f = float(dur)
                if driver_id not in best_by_driver or dur_f < best_by_driver[driver_id]:
                    best_by_driver[driver_id] = dur_f
            session_label = "FP3" if "3" in stype else "FP2"
            for driver_id, best in best_by_driver.items():
                results.append(
                    SessionResult(
                        season=season,
                        round=round_num,
                        driver_id=driver_id,
                        session_type=session_label,
                        best_lap_seconds=best,
                    )
                )
        return results
