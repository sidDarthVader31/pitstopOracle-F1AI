from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from f1ml.paths import CACHE, JOLPICA_BASE

_SESSION = requests.Session()
_SESSION.headers.update(
    {"User-Agent": "pitstopOracle-F1AI/0.1 (hobby; +https://github.com)"}
)

# Seasons whose cached responses should be bypassed on read (set during --refresh ingest).
_bypass_cache_seasons: set[int] = set()


def set_bypass_cache_seasons(seasons: set[int]) -> None:
    """Skip disk cache reads for API paths belonging to these seasons."""
    global _bypass_cache_seasons
    _bypass_cache_seasons = set(seasons)


def clear_bypass_cache_seasons() -> None:
    global _bypass_cache_seasons
    _bypass_cache_seasons = set()


def _cache_path(rel: str) -> Path:
    safe = rel.strip("/").replace("/", "_").replace("?", "_").replace("&", "_")
    return CACHE / f"{safe}.json"


def _path_season(path: str) -> int | None:
    """First path segment if it looks like a four-digit season year."""
    head = path.strip("/").split("/", 1)[0]
    if head.isdigit() and len(head) == 4:
        return int(head)
    return None


def _should_bypass_cache(path: str) -> bool:
    season = _path_season(path)
    return season is not None and season in _bypass_cache_seasons


def clear_season_cache(season: int) -> int:
    """Delete cached Jolpica JSON for one season. Returns number of files removed."""
    CACHE.mkdir(parents=True, exist_ok=True)
    prefix = f"{season}_"
    removed = 0
    for dest in CACHE.glob("*.json"):
        if dest.name.startswith(prefix):
            dest.unlink(missing_ok=True)
            removed += 1
    return removed


def clear_season_caches(seasons: range | set[int]) -> int:
    """Clear Jolpica disk cache for multiple seasons."""
    return sum(clear_season_cache(int(s)) for s in seasons)


def get_json(path: str, params: dict[str, Any] | None = None, min_interval: float = 0.3) -> dict:
    """GET Jolpica JSON with disk cache and polite rate limiting."""
    params = params or {}
    key = path + "_" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    dest = _cache_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not _should_bypass_cache(path):
        return json.loads(dest.read_text())

    url = f"{JOLPICA_BASE}/{path.lstrip('/')}"
    for attempt in range(6):
        time.sleep(min_interval)
        resp = _SESSION.get(url, params=params, timeout=60)
        if resp.status_code == 404:
            payload = {"MRData": {}}
            dest.write_text(json.dumps(payload))
            return payload
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        payload = resp.json()
        dest.write_text(json.dumps(payload))
        return payload
    raise RuntimeError(f"Rate limited fetching {url}")


def fetch_paginated(path: str, table_key: str, list_key: str, limit: int = 100) -> list[dict]:
    """Walk Ergast-style offset pagination for a nested list (e.g. Races)."""
    offset = 0
    rows: list[dict] = []
    while True:
        payload = get_json(path, {"limit": limit, "offset": offset})
        table = payload.get("MRData", {}).get(table_key, {})
        chunk = table.get(list_key, []) or []
        rows.extend(chunk)
        total = int(payload.get("MRData", {}).get("total", len(rows)))
        offset += limit
        if offset >= total or not chunk:
            break
    return rows
