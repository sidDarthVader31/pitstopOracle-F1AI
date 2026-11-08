"""Dashboard pages must read metric keys that the evaluation actually writes."""

from __future__ import annotations

import re
from pathlib import Path

from f1ml.train import _oos_summary

ROOT = Path(__file__).resolve().parents[1]
UI_FILES = [ROOT / "pages" / "4_Model_Performance.py", ROOT / "f1ml" / "ui" / "charts.py", ROOT / "f1ml" / "ui" / "components.py"]


def _written_oos_keys() -> set[str]:
    metrics = [
        "pre_hit", "pre_top3", "pre_log_loss", "pre_brier", "pre_rank_corr", "pre_finish_mae",
        "pre_prior_form_log_loss", "post_hit", "post_top3", "post_log_loss", "post_brier",
        "post_rank_corr", "post_finish_mae", "post_grid_log_loss", "post_quali_gap_log_loss",
        "post_pole_log_loss", "post_grid_hit", "post_pole_hit", "post_champ_hit",
        "dnf_pr_auc", "dnf_roc_auc", "dnf_log_loss", "dnf_brier",
    ]
    row = {k: 0.5 for k in metrics}
    row.update({"test_season": 2026, "test_round": 1, "test_date": "2026-03-08"})
    return set(_oos_summary([row], eval_year=2026))


def test_ui_reads_only_oos_keys_that_evaluation_writes():
    written = _written_oos_keys()
    read = set()
    for path in UI_FILES:
        read |= set(re.findall(r"""["'](oos_[a-z0-9_]+)["']""", path.read_text()))
    missing = sorted(read - written)
    assert not missing, f"UI reads metric keys the evaluation never writes: {missing}"
