"""Shared pieces for the six-page MVP (MVP_SPEC §4–§6).

Pure helpers (bands, stage labels, weekend state) are importable without Streamlit
running, so they are unit-tested; the ``render_*`` functions draw Streamlit widgets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from f1ml import ledger
from f1ml.schedule import Weekend, countdown_text, get_weekend

PRODUCT_NAME = "Pitstop Oracle"
UNOFFICIAL = "Unofficial — not affiliated with Formula 1"

# ---------------------------------------------------------------------------
# Chance bands (MVP_SPEC §5.2). No band above "Favourite" until G3 calibration passes.
# ---------------------------------------------------------------------------
WIN_BANDS = [(0.30, "Favourite"), (0.10, "Contender"), (0.03, "Outside chance"), (0.0, "Long shot")]
PODIUM_BANDS = [(0.60, "Likely"), (0.25, "Possible"), (0.05, "Outside chance"), (0.0, "Unlikely")]
TOP10_BANDS = [(0.60, "Likely"), (0.25, "Possible"), (0.05, "Outside chance"), (0.0, "Unlikely")]
Q3_BANDS = [(0.70, "Likely Q3"), (0.30, "Borderline"), (0.0, "Unlikely")]
BAND_BARS = {"Favourite": "●●●●", "Contender": "●●●○", "Outside chance": "●●○○", "Long shot": "●○○○",
             "Likely": "●●●●", "Possible": "●●●○", "Unlikely": "●○○○", "Likely Q3": "●●●", "Borderline": "●●○"}
WIN_BAND_COPY = {"Favourite": "Favourite", "Contender": "In contention",
                 "Outside chance": "Needs things to go their way", "Long shot": "Long shot"}


def band(p: float | None, bands: list[tuple[float, str]]) -> str:
    if p is None or pd.isna(p):
        return "—"
    for threshold, label in bands:
        if float(p) >= threshold:
            return label
    return bands[-1][1]


def band_cell(p: float | None, bands: list[tuple[float, str]]) -> str:
    label = band(p, bands)
    return f"{BAND_BARS.get(label, '')} {label}".strip()


def win_band(p):
    return band(p, WIN_BANDS)


def retirement_text(p: float | None) -> str:
    """Team retirement rate as '1 in N races' (context, not a prediction)."""
    if p is None or pd.isna(p) or p <= 0:
        return "—"
    return f"1 in {max(1, round(1 / float(p)))}"


def points_range(lo: float, hi: float) -> str:
    return f"{round(lo):d} – {round(hi):d}"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
STAGE_LABELS = {
    "pre_weekend": "Pre-weekend",
    "pre_quali": "Pre-lock forecast",
    "post_quali": "Post-qualifying",
    "grid_set": "Grid set",
    "post_sprint": "Post-sprint",
    "complete": "Race finished",
}


def display_stage(entry_or_payload: dict | None) -> str:
    if not entry_or_payload:
        return "not_published"
    stage = entry_or_payload.get("stage")
    if stage == "post_quali" and entry_or_payload.get("grid_status") == "official":
        return "grid_set"
    return stage


@dataclass
class WeekendState:
    season: int
    round: int
    weekend: Weekend | None
    entry: dict | None              # latest ledger index line for this race
    payload: dict | None            # its frozen content
    data_stage: str                 # what the data says now: pre_quali / post_quali / grid_set / post_sprint / complete
    now: datetime

    @property
    def lock_passed(self) -> bool:
        return bool(self.weekend and self.weekend.lock_at and self.now >= self.weekend.lock_at)

    @property
    def stale(self) -> str | None:
        """Warning when the data has moved on but no newer snapshot was frozen (MVP_SPEC §5.3)."""
        if self.payload is None:
            return None
        published = display_stage(self.payload)
        order = ["pre_quali", "post_quali", "grid_set", "post_sprint", "complete"]
        if self.data_stage in order and published in order and order.index(self.data_stage) > order.index(published):
            if self.data_stage == "complete":
                return None
            return (f"Qualifying data is in, but the {STAGE_LABELS[self.data_stage].lower()} forecast hasn't been "
                    f"published yet. Below is the {STAGE_LABELS[published].lower()} forecast.")
        return None


def weekend_state(season: int, round_num: int, data_stage: str, now: datetime | None = None,
                  root=None) -> WeekendState:
    now = now or datetime.now(timezone.utc)
    entry = ledger.latest_snapshot(season, round_num, root=root)
    payload = ledger.load_snapshot(entry, root) if entry else None
    return WeekendState(season, round_num, get_weekend(season, round_num), entry, payload, data_stage, now)


def next_run_text(state: WeekendState) -> str:
    wk = state.weekend
    if wk is None or wk.lock_at is None:
        return "The pre-lock forecast is published before qualifying."
    run = wk.lock_at.replace(hour=18, minute=0, second=0) - timedelta(days=1)  # runbook: 18:00 UTC the day before
    if run > wk.lock_at - timedelta(hours=1):
        run = wk.lock_at - timedelta(hours=2)
    return f"Next run: {run:%a %d %b %H:%M} UTC (team lock {wk.lock_at:%a %d %b %H:%M} UTC)."


def fmt_utc(iso: str | None) -> str:
    if not iso:
        return "—"
    ts = pd.Timestamp(iso)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%a %d %b %H:%M UTC")


def drivers_frame(payload: dict) -> pd.DataFrame:
    """Snapshot drivers as a flat table (fantasy fields prefixed fx_)."""
    rows = []
    for d in payload["drivers"]:
        row = {k: v for k, v in d.items() if k not in ("fantasy", "finish_position_probs")}
        for k, v in (d.get("fantasy") or {}).items():
            if k != "breakdown":
                row[f"fx_{k}"] = v
        row["breakdown"] = (d.get("fantasy") or {}).get("breakdown", {})
        row["name"] = f"{d.get('given_name', '')} {d.get('family_name', '')}".strip() or d["driver_id"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Streamlit widgets
# ---------------------------------------------------------------------------
def dev_mode() -> bool:
    """Developer details only with ?dev=1 in the URL (MVP_SPEC §4.1)."""
    try:
        return st.query_params.get("dev") == "1"
    except Exception:
        return False


def render_stage_chip(stage: str, extra: str = "") -> None:
    label = STAGE_LABELS.get(stage, "Not published yet")
    st.markdown(f'<span class="phase-chip">● {label}</span> <span class="po-muted">{extra}</span>',
                unsafe_allow_html=True)


def render_lock_line(state: WeekendState) -> None:
    wk = state.weekend
    if wk is None or wk.lock_at is None:
        return
    text = countdown_text(wk.lock_at, state.now)
    if text == "passed":
        st.markdown(f"**Team lock passed** ({wk.lock_at:%a %d %b %H:%M} UTC, start of qualifying).")
    else:
        st.markdown(f"**Team lock in {text}** · {wk.lock_at:%a %d %b %H:%M} UTC (start of qualifying)")


def render_freshness(payload: dict) -> None:
    parts = [f"Forecast generated {fmt_utc(payload.get('generated_at'))}",
             f"data as of {fmt_utc(payload.get('data_as_of'))}",
             f"model {payload.get('model_version', '?').replace('weekend-', 'v')}"]
    late = payload.get("published_before_lock") is False and payload.get("stage") == "pre_quali"
    st.caption(" · ".join(parts) + ("  ·  ⚠ published after the pre-lock deadline" if late else ""))


def render_limits_box() -> None:
    st.info(
        "**What this model can't do (yet).** It doesn't use practice times or weather, so rain and big setup "
        "changes surprise it. After qualifying, the starting grid is a strong guide on its own; check our record "
        "against it. It can't predict retirements; we show each team's retirement rate instead.",
        icon="ℹ️",
    )
    page_link("pages/track_record.py", "Our full record →")


def page_link(page: str, label: str, icon: str | None = None) -> None:
    """st.page_link, falling back to a plain link when a page runs outside st.navigation (tests)."""
    try:
        st.page_link(page, label=label, icon=icon)
    except st.errors.StreamlitAPIException:
        slug = page.rsplit("/", 1)[-1].removesuffix(".py")
        st.markdown(f"[{label}](/{slug})")


def render_waitlist(key: str) -> None:
    """Email capture placeholder: nothing is sent or stored until the owner picks a tool (MVP_SPEC M10)."""
    with st.form(f"waitlist_{key}", clear_on_submit=True):
        st.markdown("**Weekly pre-lock email**")
        email = st.text_input("Email", placeholder="you@example.com", label_visibility="collapsed")
        agree = st.checkbox("Send me the weekly pre-lock email. I can unsubscribe any time.")
        sent = st.form_submit_button("Join")
    if sent:
        if not agree or "@" not in (email or ""):
            st.warning("Add an email address and tick the box.")
        else:
            st.success("Thanks! Sign-ups open soon: this form is a placeholder and nothing was saved.")


def render_footer() -> None:
    st.markdown(
        f'<div class="po-footer">{UNOFFICIAL}, the FIA, any team or the F1 Fantasy game. '
        "Projections, not advice. Not betting advice. · "
        '<a href="/how_it_works" target="_self">How it works &amp; limits</a></div>',
        unsafe_allow_html=True,
    )


MVP_CSS = """
<style>
  .po-muted { color: #6b7280; font-size: 0.85rem; }
  .po-footer { color: #6b7280; font-size: 0.8rem; margin-top: 2.5rem; padding-top: 0.75rem;
               border-top: 1px solid #e5e7eb; }
  .po-hypothetical { background: #fff7ed; border: 1px dashed #c2410c; color: #9a3412; border-radius: 8px;
                     padding: 0.5rem 0.9rem; font-weight: 600; margin-bottom: 0.75rem; }
</style>
"""


def page_setup(title: str) -> None:
    from f1ml.ui.theme import apply_theme

    apply_theme()
    st.markdown(MVP_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"### {PRODUCT_NAME}")
        st.caption("Forecast and fantasy helper. " + UNOFFICIAL + ".")
