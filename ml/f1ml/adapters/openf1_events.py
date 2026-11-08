"""Historical event schedule and race-control ingestion from OpenF1."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from f1ml.paths import OPENF1_BASE, RAW


def _get_json(session: requests.Session, endpoint: str, params: dict[str, Any]) -> list[dict]:
    url = f"{OPENF1_BASE}/{endpoint}"
    for attempt in range(5):
        response = session.get(url, params=params, timeout=60)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.5 * (attempt + 1)
            except ValueError:
                delay = 2.5 * (attempt + 1)
            time.sleep(delay)
            continue
        response.raise_for_status()
        value = response.json()
        return value if isinstance(value, list) else []
    raise RuntimeError(f"OpenF1 rate limit persisted for {endpoint} {params}")


def _event_keys_for_year(year: int) -> list[tuple[int, int]]:
    results = pd.read_parquet(RAW / "results.parquet")
    year_results = results[results["season"] == year]
    return [
        (int(season), int(round_num))
        for season, round_num in year_results[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .itertuples(index=False, name=None)
    ]


def fetch_openf1_event_data(
    years: list[int],
    session: requests.Session | None = None,
    pause_seconds: float = 2.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch race sessions, meeting circuit metadata, and race-control messages.

    OpenF1's historical API starts in 2023. Race sessions are aligned to
    completed local result events in chronological order, which naturally
    skips cancelled calendar rounds and upcoming races without results.
    """
    client = session or requests.Session()
    session_rows: list[dict] = []
    meeting_rows: list[dict] = []
    control_rows: list[dict] = []
    for year in years:
        meeting_data = _get_json(client, "meetings", {"year": int(year)})
        meeting_by_key = {int(m["meeting_key"]): m for m in meeting_data if m.get("meeting_key")}
        schedule = _get_json(client, "sessions", {"year": int(year)})
        races = [
            row
            for row in schedule
            if row.get("session_type") == "Race"
            and not row.get("is_cancelled", False)
            and int(row.get("meeting_key", -1)) in meeting_by_key
        ]
        races.sort(key=lambda row: row.get("date_start", ""))
        event_keys = _event_keys_for_year(year)
        if len(races) < len(event_keys):
            raise ValueError(
                f"OpenF1 returned {len(races)} race sessions for {len(event_keys)} completed {year} events"
            )
        for event_key, race_session in zip(event_keys, races):
            meeting = meeting_by_key[int(race_session["meeting_key"])]
            session_row = {
                "season": event_key[0],
                "round": event_key[1],
                "session_key": int(race_session["session_key"]),
                "meeting_key": int(race_session["meeting_key"]),
                "session_start_utc": race_session.get("date_start"),
                "session_end_utc": race_session.get("date_end"),
                "circuit_key": meeting.get("circuit_key"),
                "circuit_short_name": meeting.get("circuit_short_name"),
                "circuit_type": meeting.get("circuit_type"),
                "circuit_info_url": meeting.get("circuit_info_url"),
                "meeting_name": meeting.get("meeting_name"),
                "source": "OpenF1",
            }
            session_rows.append(session_row)
            meeting_rows.append({"season": year, "round": event_key[1], **meeting, "source": "OpenF1"})
            time.sleep(pause_seconds)
            control = _get_json(client, "race_control", {"session_key": int(race_session["session_key"])})
            for record in control:
                control_rows.append(
                    {
                        "season": event_key[0],
                        "round": event_key[1],
                        "session_key": int(race_session["session_key"]),
                        **record,
                        "source": "OpenF1",
                    }
                )
            time.sleep(pause_seconds)
    return pd.DataFrame(session_rows), pd.DataFrame(meeting_rows), pd.DataFrame(control_rows)


def save_openf1_event_data(years: list[int]) -> dict[str, int]:
    """Fetch and atomically replace the local OpenF1 historical event tables."""
    sessions, meetings, control = fetch_openf1_event_data(years)
    RAW.mkdir(parents=True, exist_ok=True)
    for frame, name in (
        (sessions, "openf1_race_sessions.parquet"),
        (meetings, "openf1_meetings.parquet"),
        (control, "openf1_race_control.parquet"),
    ):
        frame.to_parquet(RAW / name, index=False)
    from f1ml.safety_car import build_safety_car_labels

    calendar = pd.read_parquet(RAW / "races.parquet")
    labels = build_safety_car_labels(sessions, control, calendar)
    labels.to_parquet(RAW / "openf1_safety_car_labels.parquet", index=False)
    return {
        "race_sessions": len(sessions),
        "meetings": len(meetings),
        "race_control": len(control),
        "safety_car_labels": len(labels),
    }
