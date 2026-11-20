"""Race-weekend session times and the F1 Fantasy team lock (UTC).

Source: ``data/manual/weekend_schedule.csv`` (refresh with ``scripts/sync_schedule.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from f1ml.paths import MANUAL

SCHEDULE_PATH = MANUAL / "weekend_schedule.csv"
# The pre-lock snapshot must exist at least this long before the lock (MVP_SPEC §9.1).
PRE_LOCK_MARGIN = timedelta(hours=1)


@dataclass(frozen=True)
class Weekend:
    season: int
    round: int
    race_name: str
    format: str
    quali_start: datetime | None
    sprint_quali_start: datetime | None
    race_start: datetime | None

    @property
    def lock_at(self) -> datetime | None:
        """Team lock: sprint qualifying on sprint weekends (to be confirmed), else qualifying."""
        return self.sprint_quali_start if self.format == "sprint" and self.sprint_quali_start else self.quali_start

    @property
    def is_sprint(self) -> bool:
        return self.format == "sprint"


def _parse(value) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime().astimezone(timezone.utc)


def load_schedule() -> pd.DataFrame:
    if not SCHEDULE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SCHEDULE_PATH)


def get_weekend(season: int, round_num: int, schedule: pd.DataFrame | None = None) -> Weekend | None:
    df = schedule if schedule is not None else load_schedule()
    if df.empty:
        return None
    row = df[(df["season"] == season) & (df["round"] == round_num)]
    if row.empty:
        return None
    r = row.iloc[0]
    return Weekend(
        season=int(r["season"]),
        round=int(r["round"]),
        race_name=str(r["race_name"]),
        format=str(r.get("format", "standard")),
        quali_start=_parse(r.get("quali_start_utc")),
        sprint_quali_start=_parse(r.get("sprint_quali_start_utc")),
        race_start=_parse(r.get("race_start_utc")),
    )


def countdown_text(target: datetime | None, now: datetime | None = None) -> str:
    """'1d 04h 12m' until target; 'passed' once it is in the past; '' when unknown."""
    if target is None:
        return ""
    now = now or datetime.now(timezone.utc)
    delta = target - now
    if delta.total_seconds() <= 0:
        return "passed"
    days, rem = divmod(int(delta.total_seconds()), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return f"{days}d {hours:02d}h {minutes:02d}m" if days else f"{hours}h {minutes:02d}m"
