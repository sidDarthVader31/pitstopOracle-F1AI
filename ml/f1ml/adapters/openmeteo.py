"""Open-Meteo forecast adapter for upcoming race weather."""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from f1ml.adapters.base import SourceAdapter
from f1ml.paths import RAW
from f1ml.schema.entities import CanonicalBundle, WeatherObs

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "pitstopOracle-F1AI/1.0"})


class OpenMeteoForecastAdapter(SourceAdapter):
    name = "open_meteo"

    def pull(self, season: int, round_num: int | None = None) -> CanonicalBundle:
        bundle = CanonicalBundle()
        races = pd.read_parquet(RAW / "races.parquet")
        sub = races[races["season"] == season]
        if round_num is not None:
            sub = sub[sub["round"] == round_num]
        today = date.today()
        for row in sub.itertuples(index=False):
            race_date = date.fromisoformat(str(row.date)[:10])
            if race_date < today:
                continue
            lat, lng = row.lat, row.lng
            if lat is None or lng is None:
                continue
            params = {
                "latitude": lat,
                "longitude": lng,
                "daily": "precipitation_sum,rain_sum,weathercode",
                "timezone": "UTC",
                "start_date": str(row.date)[:10],
                "end_date": str(row.date)[:10],
            }
            time.sleep(0.2)
            resp = _SESSION.get(FORECAST_API, params=params, timeout=60)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            daily = payload.get("daily", {})
            precip = (daily.get("precipitation_sum") or [None])[0]
            bundle.weather.append(
                WeatherObs(
                    season=int(row.season),
                    round=int(row.round),
                    precipitation_mm=float(precip) if precip is not None else None,
                    is_wet=bool(precip is not None and float(precip) >= 1.0),
                    forecast=True,
                    source="open_meteo_forecast",
                )
            )
        return bundle
