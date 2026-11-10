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


def test_summary_keys_are_labelled_2026_out_of_sample():
    from f1ml.train import _add_compat_aliases

    wf = [{"test_season": 2026, "post_hit": 1.0, "post_top3": 1.0, "post_pole_hit": 0.0, "post_champ_hit": 0.0}]
    metrics = {"walk_forward": wf, "oos": {"oos_post_hit": 1.0}, "train_post_quali": {}}
    _add_compat_aliases(metrics)
    assert not [k for k in metrics if k.startswith("test_") or "_rf_" in k]
    assert metrics["oos2026_post_quali"]["oos2026_post_quali_hit"] == 1.0


def test_dashboard_reads_new_and_legacy_metric_files():
    import json

    from f1ml.ui import charts

    legacy = {"test_post_quali": {"test_post_quali_hit": 0.4, "test_post_quali_log_loss": 1.6}}
    assert charts._eval_split(legacy, "post_quali")["oos2026_post_quali_hit"] == 0.4
    new = {"oos": {"oos_n_races": 19, "oos_post_n_races": 19, "oos_post_hit": 0.42}}
    assert charts._eval_split(new, "post_quali")["oos2026_post_quali_hit"] == 0.42
    assert charts._oos_label(new) == "2026 out-of-sample (19 races)"
    real = ROOT / "reports" / "metrics.json"
    if real.exists():
        fig = charts.accuracy_bars(json.loads(real.read_text()), mode="post_quali")
        assert fig.data[1].name.startswith("2026 out-of-sample")
