"""Dashboard pages must read metric keys that the evaluation actually writes."""

from __future__ import annotations

import re
from pathlib import Path

from f1ml.train import _oos_summary

ROOT = Path(__file__).resolve().parents[1]
PAGE_FILES = sorted((ROOT / "pages").glob("*.py"))
UI_FILES = PAGE_FILES + [ROOT / "app.py"] + sorted((ROOT / "f1ml" / "ui").glob("*.py"))


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


# ---------------------------------------------------------------------------
# MVP copy and structure rules (MVP_SPEC §1.2, §4, §5)
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

MVP_PAGES = ["home.py", "fantasy_helper.py", "track_record.py", "how_it_works.py", "scenario_lab.py", "race_explorer.py"]
BETTING_WORDS = re.compile(
    r"\b(odds|bets?|betting|bookies?|bookmakers?|wager|stake|banker|edge|value bets?|tips?|tipster|markets?|"
    r"beat the (?:bookies|grid|experts))\b", re.IGNORECASE)
ALLOWED_PHRASES = ("not betting advice", "no betting")


def _visible_copy(path: Path) -> str:
    """String literals only (code identifiers like market_probs are not user-facing copy)."""
    import ast

    tree = ast.parse(path.read_text())
    text = " ".join(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    low = text.lower()
    for phrase in ALLOWED_PHRASES:
        low = low.replace(phrase, "")
    return low


def test_app_has_exactly_the_six_mvp_pages():
    assert sorted(p.name for p in PAGE_FILES) == sorted(MVP_PAGES)
    app = (ROOT / "app.py").read_text()
    for name in MVP_PAGES:
        assert f"pages/{name}" in app


@pytest.mark.parametrize("name", MVP_PAGES + ["../app.py", "../f1ml/ui/mvp.py"])
def test_no_betting_language_in_user_facing_copy(name):
    path = (ROOT / "pages" / name).resolve()
    found = BETTING_WORDS.findall(_visible_copy(path))
    assert not found, f"betting language in {path.name}: {found}"


@pytest.mark.parametrize("name", MVP_PAGES)
def test_every_page_has_the_unofficial_footer_and_no_developer_controls(name):
    src = (ROOT / "pages" / name).read_text()
    assert "render_footer()" in src
    assert "sidebar_model_controls" not in src and "Normalize" not in src
    assert "Win %" not in src and "Podium %" not in src  # bands, not percentages, before G3


def test_footer_says_unofficial():
    from f1ml.ui import mvp

    assert mvp.UNOFFICIAL == "Unofficial — not affiliated with Formula 1"


def test_chance_bands_follow_the_spec():
    from f1ml.ui.mvp import PODIUM_BANDS, Q3_BANDS, WIN_BANDS, band, retirement_text

    assert [band(p, WIN_BANDS) for p in (0.62, 0.30, 0.29, 0.10, 0.05, 0.02)] == [
        "Favourite", "Favourite", "Contender", "Contender", "Outside chance", "Long shot"]
    assert "Strong favourite" not in {b for _, b in WIN_BANDS}  # no band above Favourite before G3
    assert [band(p, PODIUM_BANDS) for p in (0.7, 0.3, 0.1, 0.01)] == ["Likely", "Possible", "Outside chance", "Unlikely"]
    assert band(0.8, Q3_BANDS) == "Likely Q3" and band(0.5, Q3_BANDS) == "Borderline"
    assert retirement_text(0.125) == "1 in 8"


def test_stale_stage_warning():
    from datetime import datetime, timezone

    from f1ml.ui.mvp import WeekendState

    now = datetime(2026, 11, 21, 6, tzinfo=timezone.utc)
    st_pre = WeekendState(2026, 21, None, {}, {"stage": "pre_quali"}, "post_quali", now)
    assert st_pre.stale and "post-qualifying" in st_pre.stale
    assert WeekendState(2026, 21, None, {}, {"stage": "post_quali"}, "post_quali", now).stale is None


def test_empty_state_next_run_text():
    from datetime import datetime, timezone

    from f1ml.schedule import Weekend
    from f1ml.ui.mvp import WeekendState, next_run_text

    lock = datetime(2026, 11, 21, 4, 0, tzinfo=timezone.utc)
    wk = Weekend(2026, 21, "Las Vegas Grand Prix", "standard", lock, None, None)
    text = next_run_text(WeekendState(2026, 21, wk, None, None, "pre_quali", datetime(2026, 11, 10, tzinfo=timezone.utc)))
    assert "Fri 20 Nov 18:00 UTC" in text and "Sat 21 Nov 04:00 UTC" in text
    early = Weekend(2026, 22, "Qatar", "standard", datetime(2026, 11, 28, 18, 0, tzinfo=timezone.utc), None, None)
    assert "Fri 27 Nov 18:00" in next_run_text(WeekendState(2026, 22, early, None, None, "pre_quali", lock))
    assert "before qualifying" in next_run_text(WeekendState(2026, 22, None, None, None, "pre_quali", lock))
