"""Canonical entity definitions for Pitstop Oracle.

All adapters map vendor data into these shapes. Feature engineering reads
canonical tables only — never vendor-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Driver:
    driver_id: str
    given_name: str | None = None
    family_name: str | None = None
    driver_code: str | None = None
    driver_number: int | None = None
    nationality: str | None = None


@dataclass
class Constructor:
    constructor_id: str
    constructor_name: str | None = None
    nationality: str | None = None


@dataclass
class Circuit:
    circuit_id: str
    circuit_name: str | None = None
    country: str | None = None
    locality: str | None = None
    lat: float | None = None
    lng: float | None = None


@dataclass
class Event:
    """One Grand Prix weekend (season + round)."""

    season: int
    round: int
    race_name: str | None = None
    date: str | None = None
    circuit_id: str | None = None
    has_sprint: bool = False


@dataclass
class Session:
    event_season: int
    event_round: int
    session_type: str  # FP1, FP2, FP3, Qualifying, Sprint, Race
    date: str | None = None


@dataclass
class SessionResult:
    season: int
    round: int
    driver_id: str
    session_type: str
    position: int | None = None
    best_lap_seconds: float | None = None
    constructor_id: str | None = None


@dataclass
class StartingGrid:
    """Published race starting grid (penalties applied)."""

    season: int
    round: int
    driver_id: str
    grid_position: int
    quali_position: int | None = None
    source: str = "unknown"
    as_of: str | None = None


@dataclass
class PenaltyEvent:
    season: int
    round: int
    driver_id: str
    places_dropped: int
    reason: str | None = None
    as_of: str | None = None


@dataclass
class WeatherObs:
    season: int
    round: int
    precipitation_mm: float | None = None
    is_wet: bool = False
    forecast: bool = False
    source: str = "open_meteo"


@dataclass
class ChampionshipStanding:
    season: int
    after_round: int
    driver_id: str
    standing_position: int | None = None
    points: float = 0.0
    wins: int = 0
    constructor_id: str | None = None


@dataclass
class CanonicalBundle:
    """Container returned by SourceAdapter.pull()."""

    drivers: list[Driver] = field(default_factory=list)
    constructors: list[Constructor] = field(default_factory=list)
    circuits: list[Circuit] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)
    session_results: list[SessionResult] = field(default_factory=list)
    starting_grids: list[StartingGrid] = field(default_factory=list)
    penalties: list[PenaltyEvent] = field(default_factory=list)
    weather: list[WeatherObs] = field(default_factory=list)
    standings: list[ChampionshipStanding] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
