"""Win-probability calibration fitted on earlier out-of-sample races (EXP-005).

Plain language: the pace model ranks drivers; the calibration decides how *confident*
the resulting probabilities are. EXP-005 found that fitting that confidence on only
the 10 races before each forecast swings wildly (very sharp after a run of
predictable races, e.g. early 2026), and that favourites were over-confident.

``fit_grid_blend`` instead fits two numbers on every earlier walk-forward race:
- ``temperature``: how spread out the Plackett-Luce probabilities are;
- ``grid_weight``: how far to lean toward the plain "starting grid" forecast
  (``eval.grid_probabilities``), in log space. Before qualifying there is no grid, so
  the blend leans toward an equal field instead.

Adopted 2026-11-10 (owner approval): the live weekend run and the walk-forward
both use ``fit_rolling_temperature`` for pre_quali and post_quali, through
``train.train_stage_bundle``. On the EXP-006 pace model the plain rolling
temperature beat the grid blend, so the blend stays available but unused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

T_GRID = np.geomspace(0.3, 10.0, 41)
GRID_WEIGHTS = np.round(np.arange(0.0, 0.95, 0.05), 2)
EPS = 1e-15
MIN_HISTORY_RACES = 10  # fewer earlier out-of-sample races -> fall back to the 10-race temperature
ROLLING_MODES = ("pre_quali", "post_quali")
HISTORY_COLUMNS = ["season", "round", "stage", "driver_id", "pace_score", "dnf_prob", "won"]


@dataclass
class RaceInputs:
    """One race as the model saw it before lights out, plus the actual winner."""

    scores: np.ndarray  # pace scores, higher = faster
    retired: np.ndarray  # (draws, drivers) boolean DNF draws
    prior: np.ndarray  # grid-baseline (post-quali) or equal-field (pre-quali) probabilities
    won: np.ndarray | None = None  # 1 for the winner, else 0


def dnf_draws(dnf_probs: np.ndarray, seed: int, draws: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dnf = np.clip(np.asarray(dnf_probs, dtype=float), 0.0, 1.0)
    retired = rng.random((draws, len(dnf))) < dnf[None, :]
    all_out = retired.all(axis=1)
    if all_out.any():
        retired[all_out, int(np.argmin(dnf))] = False
    return retired


def pl_win_probs(scores: np.ndarray, retired: np.ndarray, temperature: float) -> np.ndarray:
    """Exact Plackett-Luce win probability averaged over DNF draws (as ``ranking.py``)."""
    adj = np.asarray(scores, dtype=float) / max(float(temperature), 0.05)
    logits = np.where(retired, -np.inf, adj[None, :])
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return (w / w.sum(axis=1, keepdims=True)).mean(axis=0)


def blend(p: np.ndarray, prior: np.ndarray, weight: float) -> np.ndarray:
    """Geometric blend p^(1-w) * prior^w, renormalised to sum to 1."""
    lp = (1 - weight) * np.log(np.clip(p, EPS, 1.0)) + weight * np.log(np.clip(prior, EPS, 1.0))
    q = np.exp(lp - lp.max())
    return q / q.sum()


def calibrated_win_probs(race: RaceInputs, temperature: float, grid_weight: float) -> np.ndarray:
    return blend(pl_win_probs(race.scores, race.retired, temperature), race.prior, grid_weight)


def _race_ll(p: np.ndarray, won: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-np.log(p[np.asarray(won) == 1][0] / p.sum()))


def loss_table(races: list[RaceInputs]) -> np.ndarray:
    """Log-loss of every race at every (temperature, grid weight): shape (T, W, races)."""
    out = np.empty((len(T_GRID), len(GRID_WEIGHTS), len(races)))
    for k, race in enumerate(races):
        for t, temp in enumerate(T_GRID):
            p = pl_win_probs(race.scores, race.retired, temp)
            for w, weight in enumerate(GRID_WEIGHTS):
                out[t, w, k] = _race_ll(blend(p, race.prior, weight), race.won)
    return out


def best_params(table: np.ndarray) -> tuple[float, float]:
    """(temperature, grid_weight) with the lowest mean log-loss over the table's races."""
    mean = table.mean(axis=2)
    t, w = np.unravel_index(int(np.argmin(mean)), mean.shape)
    return float(T_GRID[t]), float(GRID_WEIGHTS[w])


def fit_grid_blend(history: list[RaceInputs]) -> tuple[float, float]:
    """Fit on earlier out-of-sample races only (each needs ``won``)."""
    if not history:
        return 1.0, 0.0
    return best_params(loss_table(history))


def races_from_history(history: pd.DataFrame) -> list[RaceInputs]:
    """Turn stored out-of-sample rows (``HISTORY_COLUMNS``) into per-race inputs."""
    races = []
    for (season, rnd), g in history.groupby(["season", "round"], sort=True):
        won = g["won"].fillna(0).astype(int).to_numpy()
        if won.sum() != 1:
            continue
        n = len(g)
        retired = dnf_draws(g["dnf_prob"].to_numpy(float), int(season) * 100 + int(rnd))
        races.append(RaceInputs(g["pace_score"].to_numpy(float), retired, np.full(n, 1.0 / n), won))
    return races


def fit_rolling_temperature(history: pd.DataFrame | None) -> float | None:
    """Temperature with the lowest winner log-loss over every earlier out-of-sample race.

    ``history`` must hold only races before the one being forecast, for one stage.
    Returns None when there are fewer than ``MIN_HISTORY_RACES`` usable races.
    """
    if history is None or history.empty:
        return None
    races = races_from_history(history)
    if len(races) < MIN_HISTORY_RACES:
        return None
    losses = np.array([[_race_ll(pl_win_probs(r.scores, r.retired, t), r.won) for r in races] for t in T_GRID])
    return float(T_GRID[int(np.argmin(losses.mean(axis=1)))])


def history_rows(scored: pd.DataFrame, stage: str) -> pd.DataFrame:
    """The rows of a scored race to keep as calibration history (needs ``won``)."""
    out = scored.reindex(columns=HISTORY_COLUMNS).copy()
    out["stage"] = stage
    return out


# ----------------------------------------------------------------------------- EXP-008
# Points-finish calibration. One Plackett-Luce temperature is fitted to the *winner*,
# which leaves the midfield order too predictable (EXP-007): drivers given <1% to score
# points sometimes did. The sequential sampler below keeps the winner pick at the
# fitted temperature (so win probabilities are unchanged) and draws every later
# position with a softer temperature: ``temperature * rest_multiplier``.
REST_MULTIPLIERS = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])


def position_multipliers(n: int, rest_multiplier: float, schedule: str = "step") -> np.ndarray:
    """Temperature multiplier for each finishing position 1..n.

    ``step``: 1 for the winner, ``rest_multiplier`` for P2 onwards.
    ``ramp``: rises linearly from 1 (P1) to ``rest_multiplier`` (P10), flat after.
    """
    k = np.arange(n, dtype=float)
    if schedule == "step":
        return np.where(k == 0, 1.0, float(rest_multiplier))
    if schedule == "ramp":
        return 1.0 + (float(rest_multiplier) - 1.0) * np.minimum(1.0, k / 9.0)
    raise ValueError(f"unknown schedule {schedule!r}")


def sequential_pl_ranks(
    scores: np.ndarray,
    retired: np.ndarray,
    temperature: float,
    rest_multiplier: float = 1.0,
    *,
    schedule: str = "step",
    seed: int = 0,
) -> np.ndarray:
    """Sample classified finishing orders position by position (Plackett-Luce).

    ``retired`` is a (draws, drivers) boolean array of DNF draws. Returns ranks with
    the same shape: 1..k for classified drivers, 0 for retirements. With
    ``rest_multiplier=1`` this is the ordinary Plackett-Luce model used in
    ``ranking.simulate_race_draws``.
    """
    scores = np.asarray(scores, dtype=float)
    draws, n = retired.shape
    rng = np.random.default_rng(seed)
    avail = ~retired.copy()
    ranks = np.zeros((draws, n), dtype=int)
    rows = np.arange(draws)
    mults = position_multipliers(n, rest_multiplier, schedule)
    for k in range(n):
        live = avail.any(axis=1)
        if not live.any():
            break
        t = max(float(temperature) * float(mults[k]), 0.05)
        logits = scores[None, :] / t + rng.gumbel(size=(draws, n))
        logits[~avail] = -np.inf
        pick = np.argmax(logits, axis=1)
        r, p = rows[live], pick[live]
        ranks[r, p] = k + 1
        avail[r, p] = False
    return ranks


def outcome_probs_from_ranks(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Unconditional podium / points-finish probabilities and conditional expected finish."""
    classified = ranks > 0
    n_cls = classified.sum(axis=0)
    return {
        "podium_prob": ((ranks >= 1) & (ranks <= 3)).mean(axis=0),
        "points_prob": ((ranks >= 1) & (ranks <= 10)).mean(axis=0),
        "expected_finish": np.divide(
            np.where(classified, ranks, 0).sum(axis=0), n_cls,
            out=np.full(ranks.shape[1], np.nan), where=n_cls > 0,
        ),
    }


def binary_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_rest_multiplier(per_race_losses: np.ndarray, min_races: int = MIN_HISTORY_RACES) -> float | None:
    """Pick the multiplier with the lowest mean points log-loss over earlier races.

    ``per_race_losses``: shape (len(REST_MULTIPLIERS), races), earlier races only.
    Returns None with fewer than ``min_races`` races (keep the multiplier at 1).
    """
    if per_race_losses.ndim != 2 or per_race_losses.shape[1] < min_races:
        return None
    return float(REST_MULTIPLIERS[int(np.argmin(per_race_losses.mean(axis=1)))])
