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

Nothing here is wired into the live weekend models yet; changing the approved model
is the owner's call (see docs/experiments/EXP-005-calibration.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_GRID = np.geomspace(0.3, 10.0, 41)
GRID_WEIGHTS = np.round(np.arange(0.0, 0.95, 0.05), 2)
EPS = 1e-15


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
