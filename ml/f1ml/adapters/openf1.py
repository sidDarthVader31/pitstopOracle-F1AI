"""OpenF1 adapter — starting grid, practice pace, and race pace."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from f1ml.adapters.base import SourceAdapter
from f1ml.calendar import next_upcoming_race
from f1ml.paths import CACHE, CANONICAL, RAW
from f1ml.schema.entities import CanonicalBundle, SessionResult, StartingGrid

OPENF1_BASE = "https://api.openf1.org/v1"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "pitstopOracle-F1AI/1.0"})

FP_SESSIONS = ("Practice 1", "Practice 2", "Practice 3", "FP1", "FP2", "FP3")
PACE_SESSION_MAP = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "FP1": "FP1",
    "FP2": "FP2",
    "FP3": "FP3",
    "Race": "Race",
}


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


def _plain(text: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c)).lower().strip()


def _augment_number_map(num_map: dict[int, str], session_key: int, season: int) -> dict[int, str]:
    """Fill driver numbers Jolpica does not know (e.g. debutants) from OpenF1's driver list.

    Matches OpenF1 name acronym to Jolpica driver code, then accent-insensitive last name.
    """
    out = dict(num_map)
    drivers = _get_json("drivers", {"session_key": session_key})
    missing = [d for d in drivers if d.get("driver_number") is not None and int(d["driver_number"]) not in out]
    if not missing:
        return out
    frames = [pd.read_parquet(RAW / n) for n in ("qualifying.parquet", "results.parquet") if (RAW / n).exists()]
    if not frames:
        return out
    roster = pd.concat(frames, ignore_index=True)
    roster = roster[roster["season"] == season].drop_duplicates("driver_id")
    by_code = {str(c): d for c, d in zip(roster["driver_code"], roster["driver_id"]) if pd.notna(c)}
    by_name = {_plain(n): d for n, d in zip(roster["family_name"], roster["driver_id"]) if pd.notna(n)}
    taken = set(out.values())
    for d in missing:
        driver_id = by_code.get(str(d.get("name_acronym"))) or by_name.get(_plain(d.get("last_name", "")))
        if driver_id and driver_id not in taken:
            out[int(d["driver_number"])] = driver_id
            taken.add(driver_id)
    return out


def _meeting_key_for_round(season: int, round_num: int) -> int | None:
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
    return int(meeting_key) if meeting_key is not None else None


def _session_lap_stats(
    session_key: int,
    num_map: dict[int, str],
    session_label: str,
    season: int,
    round_num: int,
) -> list[SessionResult]:
    laps = _get_json("laps", {"session_key": session_key})
    if not laps:
        return []
    by_driver: dict[str, list[float]] = {}
    for lap in laps:
        dur = lap.get("lap_duration")
        num = lap.get("driver_number")
        if dur is None or num is None:
            continue
        driver_id = num_map.get(int(num))
        if not driver_id:
            continue
        dur_f = float(dur)
        if dur_f <= 0 or dur_f > 200:
            continue
        by_driver.setdefault(driver_id, []).append(dur_f)

    results: list[SessionResult] = []
    for driver_id, times in by_driver.items():
        times_sorted = sorted(times)
        best = times_sorted[0]
        longrun = float(np_median(times_sorted[1:21])) if len(times_sorted) > 1 else best
        results.append(
            SessionResult(
                season=season,
                round=round_num,
                driver_id=driver_id,
                session_type=session_label,
                best_lap_seconds=best,
            )
        )
        # Store long-run as extra via session_type suffix in parquet writer.
        results.append(
            SessionResult(
                season=season,
                round=round_num,
                driver_id=driver_id,
                session_type=f"{session_label}_LONGRUN",
                best_lap_seconds=longrun,
            )
        )
    return results


def np_median(vals: list[float]) -> float:
    import numpy as np

    return float(np.median(vals))


class OpenF1Adapter(SourceAdapter):
    name = "openf1"

    def pull(self, season: int, round_num: int | None = None) -> CanonicalBundle:
        bundle = CanonicalBundle()
        if round_num is not None:
            rounds = [round_num]
        else:
            try:
                s, upcoming = next_upcoming_race()
                rounds = [upcoming] if s == season else []
            except Exception:
                rounds = []
        for rnd in rounds:
            bundle.starting_grids.extend(self._fetch_starting_grid(season, int(rnd)))
            bundle.session_results.extend(self._fetch_session_pace(season, int(rnd)))
        return bundle

    def _fetch_starting_grid(self, season: int, round_num: int) -> list[StartingGrid]:
        meeting_key = _meeting_key_for_round(season, round_num)
        if meeting_key is None:
            return []
        sessions = _get_json("sessions", {"meeting_key": meeting_key})
        # OpenF1 publishes the Grand Prix starting grid (with penalties applied)
        # under the *Qualifying* session, not the Race session.
        session_key = None
        for s in sessions:
            if s.get("session_name") == "Qualifying":
                session_key = int(s["session_key"])
                break
        if session_key is None:
            return []
        rows = _get_json("starting_grid", {"session_key": session_key})
        if not rows:
            return []
        num_map = _augment_number_map(_driver_number_map(season), session_key, season)
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

    def _fetch_session_pace(self, season: int, round_num: int) -> list[SessionResult]:
        meeting_key = _meeting_key_for_round(season, round_num)
        if meeting_key is None:
            return []
        sessions = _get_json("sessions", {"meeting_key": meeting_key})
        num_map = _driver_number_map(season)
        if sessions and sessions[0].get("session_key"):
            num_map = _augment_number_map(num_map, int(sessions[0]["session_key"]), season)
        results: list[SessionResult] = []
        for sess in sessions:
            stype = sess.get("session_name") or sess.get("session_type") or ""
            label = PACE_SESSION_MAP.get(stype)
            if label is None:
                continue
            session_key = sess.get("session_key")
            if not session_key:
                continue
            results.extend(
                _session_lap_stats(int(session_key), num_map, label, season, round_num)
            )
        return results


def _session_results_to_frame(results: list[SessionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "season": r.season,
                "round": r.round,
                "driver_id": r.driver_id,
                "session_type": r.session_type,
                "best_lap_seconds": r.best_lap_seconds,
            }
        )
    return pd.DataFrame(rows)


def backfill_fp_pace(
    start_year: int,
    end_year: int,
    *,
    max_rounds_per_season: int | None = None,
    skip_existing: bool = True,
    force_rounds: set[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Backfill FP/Race pace from OpenF1 for all completed rounds."""
    races = pd.read_parquet(RAW / "races.parquet")
    existing_path = CANONICAL / "fp_pace.parquet"
    existing = pd.DataFrame()
    if skip_existing and existing_path.exists():
        existing = pd.read_parquet(existing_path)

    force_rounds = force_rounds or set()
    pace_sessions = ("FP1", "FP2", "FP3", "Race")
    adapter = OpenF1Adapter()
    all_results: list[SessionResult] = []
    for year in range(start_year, end_year + 1):
        season_races = races[races["season"] == year].sort_values("round")
        if max_rounds_per_season:
            season_races = season_races.head(max_rounds_per_season)
        for _, race in season_races.iterrows():
            rnd = int(race["round"])
            if (year, rnd) in force_rounds:
                # Replace stored rows only if the refetch returns data: OpenF1 rate
                # limits come back as an empty list, which must not wipe good data.
                print(f"OpenF1 pace: {year} R{rnd:02d} (refresh) ...")
                fresh = adapter._fetch_session_pace(year, rnd)
                if fresh and not existing.empty:
                    existing = existing[
                        ~((existing["season"] == year) & (existing["round"] == rnd))
                    ]
                all_results.extend(fresh)
                continue
            elif skip_existing and not existing.empty:
                have = existing[
                    (existing["season"] == year)
                    & (existing["round"] == rnd)
                    & (existing["session_type"].isin(pace_sessions))
                ]
                if len(have) >= 10:
                    continue
            print(f"OpenF1 pace: {year} R{rnd:02d} ...")
            all_results.extend(adapter._fetch_session_pace(year, rnd))

    new_df = _session_results_to_frame(all_results)
    if existing.empty:
        merged = new_df
    elif new_df.empty:
        merged = existing
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            ["season", "round", "driver_id", "session_type"], keep="last"
        )
    CANONICAL.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(existing_path, index=False)
    return merged


def write_fp_pace_from_bundle(bundle: CanonicalBundle) -> None:
    if not bundle.session_results:
        return
    path = CANONICAL / "fp_pace.parquet"
    new_df = _session_results_to_frame(bundle.session_results)
    if new_df.empty:
        return
    rounds = new_df[["season", "round"]].drop_duplicates()
    if path.exists():
        existing = pd.read_parquet(path)
        for _, row in rounds.iterrows():
            existing = existing[
                ~(
                    (existing["season"] == row["season"])
                    & (existing["round"] == row["round"])
                )
            ]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.drop_duplicates(
        ["season", "round", "driver_id", "session_type"], keep="last"
    )
    CANONICAL.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
