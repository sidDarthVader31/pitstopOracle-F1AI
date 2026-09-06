from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import requests

from datetime import date

from f1ml.jolpica import (
    clear_bypass_cache_seasons,
    clear_season_caches,
    fetch_paginated,
    get_json,
    set_bypass_cache_seasons,
)
from f1ml.paths import CACHE, END_YEAR, OPEN_METEO, RAW, START_YEAR

_WEATHER = requests.Session()
_WEATHER.headers.update({"User-Agent": "pitstopOracle-F1AI/0.1"})


def _parse_time_to_seconds(value: str | None) -> float | None:
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None
    return None


def _driver_row(driver: dict) -> dict[str, Any]:
    return {
        "driver_id": driver.get("driverId"),
        "driver_code": driver.get("code"),
        "driver_number": driver.get("permanentNumber"),
        "given_name": driver.get("givenName"),
        "family_name": driver.get("familyName"),
        "nationality": driver.get("nationality"),
    }


def _constructor_row(constructor: dict) -> dict[str, Any]:
    return {
        "constructor_id": constructor.get("constructorId"),
        "constructor_name": constructor.get("name"),
        "constructor_nationality": constructor.get("nationality"),
    }


def _circuit_row(circuit: dict) -> dict[str, Any]:
    loc = circuit.get("Location", {})
    return {
        "circuit_id": circuit.get("circuitId"),
        "circuit_name": circuit.get("circuitName"),
        "locality": loc.get("locality"),
        "country": loc.get("country"),
        "lat": float(loc["lat"]) if loc.get("lat") else None,
        "lng": float(loc["long"]) if loc.get("long") else None,
    }


def ingest_races(years: range) -> pd.DataFrame:
    rows = []
    for year in years:
        races = fetch_paginated(f"{year}/races.json", "RaceTable", "Races")
        for race in races:
            row = {
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race.get("raceName"),
                "date": race.get("date"),
                "time": race.get("time"),
            }
            row.update(_circuit_row(race.get("Circuit", {})))
            rows.append(row)
    return pd.DataFrame(rows)


def ingest_results(years: range) -> pd.DataFrame:
    rows = []
    for year in years:
        races = fetch_paginated(f"{year}/results.json", "RaceTable", "Races")
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            for res in race.get("Results", []) or []:
                row = {
                    "season": season,
                    "round": rnd,
                    "position": int(res["position"]) if str(res.get("position", "")).isdigit() else None,
                    "position_text": res.get("positionText"),
                    "points": float(res.get("points", 0)),
                    "grid": int(res["grid"]) if str(res.get("grid", "")).isdigit() else None,
                    "laps": int(res["laps"]) if str(res.get("laps", "")).isdigit() else None,
                    "status": res.get("status"),
                    "time": (res.get("Time") or {}).get("time"),
                }
                row.update(_driver_row(res.get("Driver", {})))
                row.update(_constructor_row(res.get("Constructor", {})))
                rows.append(row)
    return pd.DataFrame(rows)


def ingest_qualifying(years: range) -> pd.DataFrame:
    rows = []
    for year in years:
        races = fetch_paginated(f"{year}/qualifying.json", "RaceTable", "Races")
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            for res in race.get("QualifyingResults", []) or []:
                q1, q2, q3 = res.get("Q1"), res.get("Q2"), res.get("Q3")
                best = next((t for t in (q3, q2, q1) if t), None)
                row = {
                    "season": season,
                    "round": rnd,
                    "quali_position": int(res["position"]) if str(res.get("position", "")).isdigit() else None,
                    "q1": q1,
                    "q2": q2,
                    "q3": q3,
                    "q1_seconds": _parse_time_to_seconds(q1),
                    "q2_seconds": _parse_time_to_seconds(q2),
                    "q3_seconds": _parse_time_to_seconds(q3),
                    "best_quali_seconds": _parse_time_to_seconds(best),
                }
                row.update(_driver_row(res.get("Driver", {})))
                row.update(_constructor_row(res.get("Constructor", {})))
                rows.append(row)
    return pd.DataFrame(rows)


def ingest_sprints(years: range) -> pd.DataFrame:
    rows = []
    for year in years:
        races = fetch_paginated(f"{year}/sprint.json", "RaceTable", "Races")
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            for res in race.get("SprintResults", []) or []:
                row = {
                    "season": season,
                    "round": rnd,
                    "sprint_position": int(res["position"]) if str(res.get("position", "")).isdigit() else None,
                    "sprint_points": float(res.get("points", 0)),
                    "sprint_grid": int(res["grid"]) if str(res.get("grid", "")).isdigit() else None,
                    "sprint_status": res.get("status"),
                }
                row.update(_driver_row(res.get("Driver", {})))
                rows.append(row)
    return pd.DataFrame(rows)


def ingest_standings_after_rounds(races: pd.DataFrame) -> pd.DataFrame:
    """Standings *after* each round; used as pre-race standings for the next round."""
    rows = []
    for season, rnd in races[["season", "round"]].drop_duplicates().itertuples(index=False):
        payload = get_json(f"{season}/{rnd}/driverStandings.json", {"limit": 100})
        lists = (
            payload.get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [])
            or []
        )
        for standing_list in lists:
            for item in standing_list.get("DriverStandings", []) or []:
                row = {
                    "season": int(season),
                    "after_round": int(rnd),
                    "standing_position": int(item["position"]) if str(item.get("position", "")).isdigit() else None,
                    "points": float(item.get("points", 0)),
                    "wins": int(item.get("wins", 0)),
                }
                row.update(_driver_row(item.get("Driver", {})))
                constructors = item.get("Constructors") or []
                if constructors:
                    row.update(_constructor_row(constructors[0]))
                rows.append(row)
    return pd.DataFrame(rows)


def ingest_weather(races: pd.DataFrame) -> pd.DataFrame:
    rows = []
    today = date.today()
    grouped = races.dropna(subset=["lat", "lng", "date"]).drop_duplicates(
        ["season", "round", "lat", "lng", "date"]
    )
    for rec in grouped.itertuples(index=False):
        race_date = date.fromisoformat(str(rec.date)[:10])
        dest = CACHE / f"weather_{rec.season}_{rec.round}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)

        if race_date > today:
            payload = {}
        elif dest.exists():
            payload = json.loads(dest.read_text())
        else:
            params = {
                "latitude": rec.lat,
                "longitude": rec.lng,
                "start_date": rec.date,
                "end_date": rec.date,
                "daily": "precipitation_sum,rain_sum,weathercode",
                "timezone": "UTC",
            }
            time.sleep(0.2)
            resp = _WEATHER.get(OPEN_METEO, params=params, timeout=60)
            if resp.status_code == 400:
                payload = {}
            else:
                resp.raise_for_status()
                payload = resp.json()
                dest.write_text(json.dumps(payload))

        daily = payload.get("daily", {})
        precip = (daily.get("precipitation_sum") or [None])[0]
        rain = (daily.get("rain_sum") or [None])[0]
        code = (daily.get("weathercode") or [None])[0]
        rows.append(
            {
                "season": rec.season,
                "round": rec.round,
                "precipitation_mm": precip,
                "rain_mm": rain,
                "weathercode": code,
                "is_wet": bool(precip is not None and precip >= 1.0),
            }
        )
    return pd.DataFrame(rows)


def run(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    *,
    refresh: bool = False,
) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    years = range(start_year, end_year + 1)
    if refresh:
        removed = clear_season_caches(years)
        set_bypass_cache_seasons(set(years))
        print(f"refresh: cleared {removed} cached Jolpica responses for {start_year}–{end_year}")
    races = ingest_races(years)
    results = ingest_results(years)
    quali = ingest_qualifying(years)
    sprints = ingest_sprints(years)
    standings = ingest_standings_after_rounds(races)
    weather = ingest_weather(races)

    races.to_parquet(RAW / "races.parquet", index=False)
    results.to_parquet(RAW / "results.parquet", index=False)
    quali.to_parquet(RAW / "qualifying.parquet", index=False)
    sprints.to_parquet(RAW / "sprints.parquet", index=False)
    standings.to_parquet(RAW / "standings.parquet", index=False)
    weather.to_parquet(RAW / "weather.parquet", index=False)

    clear_bypass_cache_seasons()
    print(f"races={len(races)} results={len(results)} quali={len(quali)} "
          f"sprints={len(sprints)} standings={len(standings)} weather={len(weather)}")
