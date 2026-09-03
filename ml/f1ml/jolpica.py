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


def _cache_path(rel: str) -> Path:
    safe = rel.strip("/").replace("/", "_").replace("?", "_").replace("&", "_")
    return CACHE / f"{safe}.json"


def get_json(path: str, params: dict[str, Any] | None = None, min_interval: float = 0.3) -> dict:
    """GET Jolpica JSON with disk cache and polite rate limiting."""
    params = params or {}
    key = path + "_" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    dest = _cache_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
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
