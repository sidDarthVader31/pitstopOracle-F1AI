# Pitstop Oracle: 2027 launch roadmap

**Owner:** Siddharth Bisht. **Coordinator:** the main Claude session.
**Created:** 2026-11-09. **Target launch:** the 2027 season opener (around mid-March 2027).

Every workstream session reads this file first. When a session finishes, it adds a dated entry to the **Decision log** at the bottom.

## Product direction (decided 2026-11-09)
- **First market:** UK/EU bettors, plus F1 Fantasy players. We sell forecasts and insights only. We never take bets.
- **UI:** Streamlit until both the features and the model are launch-worthy. After launch we build a real web app (FastAPI + React) with accounts and payments.
- **Existing product vision:** the PRD, ERD and ML Design v3 PDFs in `docs/v2/`. Markdown copies will be added at `docs/product/PRD_v3.md` and the matching files, once the owner exports the text.
- **The question that decides positioning:** bettors compare us with **bookmaker and Betfair odds**, not with the starting grid. If our probabilities are at least as accurate as the market's (with the bookmaker margin removed), we can offer "value" insights. If they are not, we position as a fantasy and race-insight tool.

## Where the model stands (from `ml/reports/EVAL.md`, 91 out-of-sample races, 2022–2026)
Log-loss scores how good the probabilities are; lower is better.

| Forecast | Model log-loss | Baseline | Notes |
|---|---|---|---|
| Post-quali winner | 1.422 | grid 1.466 | Only just beats the grid. Winner hit rate is 48.4%. |
| Post-quali winner, 2026 only | 1.618 | grid 1.609 | Slightly worse than the grid in the new regulation era. |
| Pre-quali winner | 1.788 | prior-form 2.386 | A clear gain over the prior-form baseline. |
| DNF | ROC-AUC 0.55 | 0.50 is a coin flip | Not good enough to claim as a strength. |

We have never compared the model with the betting market.

## Workstreams
| ID | Workstream | Wave | Session | Output |
|---|---|---|---|---|
| A | Market research (UK/EU bettors, fantasy, competitors, regulation, pricing) | 1 (now) | _tbd_ | `docs/product/MARKET_RESEARCH.md` |
| B | Model-vs-market benchmark + integrity re-audit | 1 (now) | _tbd_ | `docs/experiments/EXP-004-market-benchmark.md`, `docs/ML_LAUNCH_READINESS.md` |
| — | Live shadow test: save forecasts before each remaining 2026 race | 1 | coordinator | `ml/` forecast snapshots |
| C | Product design: personas, MVP spec, Streamlit wireframes | 2 (Dec) | _tbd_ | `docs/product/MVP_SPEC.md` |
| D | ML improvement experiments, one change each | 2 (Dec–Jan) | _tbd_ | `docs/experiments/EXP-00x-*.md` |
| E | Engineering and ops: Streamlit MVP, automated weekend runs, hosting | 3 (Jan–Feb) | _tbd_ | Running app |

## Gates
| Gate | Target date | Pass condition |
|---|---|---|
| G1 Market verdict | 2026-11-30 | B's benchmark is done, and positioning is decided: value insights, or a fantasy/insight tool. |
| G2 MVP scope frozen | 2026-12-15 | The owner approves `MVP_SPEC.md`. |
| G3 Model launch-worthy | 2027-01-31 | No known leakage. Post-quali beats the grid on walk-forward. Probabilities are calibrated. A live shadow record is published. |
| G4 Launch | 2027 opener | The Streamlit MVP is live, weekend runs are automated, and the track-record page is up. |
| After launch | from 2027-04 | Plan the FastAPI + React web app, accounts and payments. |

## Working rules for every session
- The owner is an experienced backend engineer with **no ML, market-research or product-design background**. Explain those terms in plain language, using F1 analogies.
- Keep three things clearly separate: what exists today, what is proposed, and future work.
- Out-of-sample evidence comes before complexity. Change one thing per experiment.
- Work on your own branch, keep changes small and tested, and never auto-promote a model.

## Decision log
- **2026-11-09.** Chose UK/EU bettors as the first market. Streamlit stays the UI until launch, and the real web app comes after launch. Workstreams run as separate Claude sessions coordinated from the main session. Note: this branch is based on `starting-grid` (commit 75ffc3f, official starting grid), which is not merged into `main` yet.
