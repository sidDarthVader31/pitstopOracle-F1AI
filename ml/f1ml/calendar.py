"""Race calendar helpers (no dependency on predict/models)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from f1ml.paths import RAW


def load_races_calendar() -> pd.DataFrame:
    """Load Jolpica race calendar from raw parquet."""
    return pd.read_parquet(RAW / "races.parquet")


def current_season() -> int:
    """Calendar year used as the live-ingest season."""
    return date.today().year


def next_upcoming_race(calendar: pd.DataFrame | None = None) -> tuple[int, int]:
    """Next GP on or after today; if the season ended, return the last round."""
    cal = calendar if calendar is not None else load_races_calendar()
    today = date.today()
    view = cal.sort_values(["season", "round"]).copy()
    view["race_date"] = pd.to_datetime(view["date"]).dt.date
    upcoming = view[view["race_date"] >= today]
    row = upcoming.iloc[0] if not upcoming.empty else view.iloc[-1]
    return int(row["season"]), int(row["round"])
