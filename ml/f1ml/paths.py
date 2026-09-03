from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

START_YEAR = 2022
END_YEAR = 2026
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
