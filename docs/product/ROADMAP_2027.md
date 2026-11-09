# Pitstop Oracle: 2027 launch roadmap

**Owner:** Siddharth Bisht. **Coordinator:** the main Claude session.
**Created:** 2026-11-09. **Target launch:** the 2027 season opener (around mid-March 2027).

Every workstream session reads this file first. When a session finishes, it adds a dated entry to the **Decision log** at the bottom.

## Product direction (decided 2026-11-09, revised the same day after G1)
- **Product:** an **F1 analytics and fantasy helper** for engaged fans and F1 Fantasy players. **No betting features of any kind:** no odds display, no market comparison in the product, no bookmaker affiliate links, no "value" or tip language. The owner wants a product without gambling-compliance constraints.
- **Still applies:** trademark care with the "F1" and "Formula 1" names, data-source terms, and UK subscription rules (from 2027-01-01) if we ever charge.
- **Market data stays internal.** It is used only as an internal benchmark to measure model quality (EXP-004), never shown to users.
- **UI:** Streamlit until both the features and the model are launch-worthy. After launch we build a real web app (FastAPI + React) with accounts and payments.
- **Existing product vision:** the v3 PRD, ML Design and ERD, now in markdown at [`PRD_v3.md`](PRD_v3.md), [`ML_MODEL_DESIGN_v3.md`](ML_MODEL_DESIGN_v3.md) and [`ERD_DATA_MODEL_v3.md`](ERD_DATA_MODEL_v3.md). The PDF originals are in `docs/v2/`. They were written from an old README, not the current code, so read the "Status vs current code" note at the top of each.
- **Positioning (G1, resolved):** the model loses clearly to the betting market (EXP-004). That, plus the owner's decision, settles the positioning on fantasy and analytics.

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
| G1 Market verdict | 2026-11-30 | **Passed 2026-11-09.** The model loses to the market, and we chose fantasy/analytics with no betting. |
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
- **2026-11-09 (workstream A, market research).** Wrote `docs/product/MARKET_RESEARCH.md`. **Recommendation: launch as Option B, "transparent F1 analytics plus fantasy helper"**, with a public timestamped track record and a free, 18+ "market-check" view (our probabilities next to de-vigged odds, with no bet buttons and no affiliate links). Upgrade to Option A ("value-odds model") only if G1 shows our post-quali log-loss is at or below the de-vigged market. Option C ("tipster") can't be proven by March 2027. Why: F1 Fantasy is big (2.75M+ teams in 2025) and low-risk to regulate. F1 is ~0.4% of global betting turnover. Free AI predictors and odds tools (Oddschecker went free in October 2026) commoditise generic probabilities. None of the competitors we found publishes probability-quality scores or a market comparison. Open items: legal review (Gambling Commission licence not needed for advice only, to be confirmed; CAP tipster proofing; DMCC subscription rules from 2027-01-01); the owner to confirm fantasy-first marketing and no affiliate income at launch. **The final choice waits for G1.**
- **2026-11-09 (G1).** Workstream B (EXP-004, branch `claude/market-benchmark-25777b`) found that the model loses to the de-margined Polymarket prices at every stage. Post-quali log-loss is 1.485 for the model vs 1.172 for the market, over 50 races. Workstream A (branch `claude/market-research-351da2`) recommended Option B, "transparent F1 analytics plus fantasy helper". **Owner decision: adopt Option B with no betting features at all, including no odds or market-comparison view and no affiliate links, to stay free of gambling-compliance constraints.** Wording in A's doc about a "market-check view" is superseded by this entry.
- **2026-11-09.** Converted the v3 PRD, ML Design and ERD PDFs to markdown in `docs/product/`. They predate the current code; see the status note in each file.
