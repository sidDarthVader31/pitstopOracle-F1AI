from __future__ import annotations

import streamlit as st

F1_RED = "#e10600"
INK = "#1a1a1a"
MUTED = "#6b7280"
SURFACE = "#ffffff"
BORDER = "#e5e7eb"


def apply_theme() -> None:
  st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }
      .stApp {
        background-color: #f8fafc;
        color: #1a1a1a;
      }
      [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
      }
      h1, h2, h3 {
        color: #111827 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
      }
      [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #111827;
        font-weight: 600;
      }
      [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 500;
      }
      .hero-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #e10600;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }
      .hero-card h3 {
        color: #e10600;
        margin: 0 0 0.35rem 0;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }
      .hero-card p {
        color: #111827;
        margin: 0;
        font-size: 1.25rem;
        font-weight: 700;
      }
      .hero-card span {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
      }
      .status-pre { color: #b45309; font-weight: 600; }
      .status-post { color: #1d4ed8; font-weight: 600; }
      .status-grid { color: #7c3aed; font-weight: 600; }
      .status-done { color: #047857; font-weight: 600; }
      .phase-chip {
        display: inline-block;
        background: #111827;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.5rem;
      }
      .disclaimer {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 2rem;
      }
      div[data-testid="stPlotlyChart"] {
        margin-bottom: 0.75rem;
      }
      h3 {
        margin-top: 1.25rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
  )


def hero_card(title: str, name: str, subtitle: str = "") -> None:
  sub = f'<span>{subtitle}</span>' if subtitle else ""
  st.markdown(
    f'<div class="hero-card"><h3>{title}</h3><p>{name}</p>{sub}</div>',
    unsafe_allow_html=True,
  )


def status_badge(mode: str) -> None:
  css = {
    "pre_quali": "status-pre",
    "post_quali": "status-post",
    "grid_set": "status-grid",
    "complete": "status-done",
  }
  labels = {
    "pre_quali": "Pre-qualifying",
    "post_quali": "Post-qualifying",
    "grid_set": "Grid set",
    "complete": "Race complete",
  }
  st.markdown(
    f'<p class="{css.get(mode, "status-pre")}">● {labels.get(mode, mode)}</p>',
    unsafe_allow_html=True,
  )


def phase_chip(phase: str) -> None:
  labels = {
    "practice": "Practice",
    "qualifying": "Qualifying",
    "grid_set": "Grid set",
    "race": "Race",
    "complete": "Complete",
  }
  st.markdown(
    f'<span class="phase-chip">{labels.get(phase, phase)}</span>',
    unsafe_allow_html=True,
  )


def disclaimer_footer() -> None:
  st.markdown(
    '<p class="disclaimer">Pitstop Oracle — ML forecasts for learning and fantasy. Not betting advice.</p>',
    unsafe_allow_html=True,
  )
