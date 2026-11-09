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
| B | Model-vs-market benchmark + integrity re-audit | 1 (now) | done 2026-11-09 | `docs/experiments/EXP-004-market-benchmark.md`, `docs/ML_LAUNCH_READINESS.md` |
| — | Live shadow test: save forecasts before each remaining 2026 race | 1 | coordinator | `ml/` forecast snapshots |
| C | Product design: personas, MVP spec, Streamlit wireframes | 2 (Dec) | draft done 2026-11-09, awaiting G2 | `docs/product/MVP_SPEC.md` |
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
- **2026-11-09. G1 market verdict (workstream B): the model loses to the market. Position as a fantasy/race-insight tool, not "value bets".** On the same 50 races (2024 Italian GP to 2026 Mexico City GP), Polymarket's de-margined winner probabilities scored log-loss **1.172 after qualifying vs 1.485 for the model** (grid 1.563). Model minus market: +0.31, 95% CI +0.11 to +0.51. The model is better in only 16 of 50 races. Before the weekend: market 1.735 vs model 1.989 (CI +0.02 to +0.51). A 50/50 model+market blend doesn't beat the market. Backing drivers whenever the model's probability is above the market's returned about −51% ROI after qualifying. Main weakness: the model's 50%+ favourites win only 39% of the time. No usable odds exist for 2022–23 without a Betfair account or a paid feed (not signed up for: owner decision). Integrity re-audit: one evaluation leak fixed (disqualified and non-starting drivers were dropped from scored fields). The re-run walk-forward post-quali log-loss is now 1.443 vs grid 1.488, and 1.637 vs 1.617 in 2026. The DNF model (ROC-AUC 0.55) supports no launch claims. Details: `docs/experiments/EXP-004-market-benchmark.md`, `docs/ML_LAUNCH_READINESS.md`. Owner still to confirm the positioning decision at G1.
- **2026-11-09 (G1).** Workstream B (EXP-004, branch `claude/market-benchmark-25777b`) found that the model loses to the de-margined Polymarket prices at every stage. Post-quali log-loss is 1.485 for the model vs 1.172 for the market, over 50 races. Workstream A (branch `claude/market-research-351da2`) recommended Option B, "transparent F1 analytics plus fantasy helper". **Owner decision: adopt Option B with no betting features at all, including no odds or market-comparison view and no affiliate links, to stay free of gambling-compliance constraints.** Wording in A's doc about a "market-check view" is superseded by this entry.
- **2026-11-09.** Converted the v3 PRD, ML Design and ERD PDFs to markdown in `docs/product/`. They predate the current code; see the status note in each file.
- **2026-11-09 (workstream C, product design). MVP scope proposed for G2 approval: [`MVP_SPEC.md`](MVP_SPEC.md).** Positioning: *an unofficial F1 forecast and fantasy helper that shows its homework*. No betting, no official affiliation, no "beats the grid" or DNF-prediction claims. Personas: casual fantasy player and fantasy min-maxer (launch audience), data-curious fan (secondary). The **pre-lock forecast** (frozen before qualifying; before sprint qualifying on sprint weekends, to be confirmed) and **projected fantasy points with a range** are the core outputs. **Must (launch):** weekend hub with lock countdown and stage/freshness labels; contenders with chance bands; driver and constructor projected points; "Check my team" with suggested transfers and 2X pick; public "Oracle team"; post-quali update; public timestamped Track Record (incl. projected vs actual fantasy points); How it works and limits; email waitlist; reliability basics. **Should:** Scenario Lab (what-ifs split out, labelled hypothetical), Race Explorer, CSV export, sprint view. **Later:** chip planner, price predictor, accounts, leagues, paid tier. PRD conflict resolved: page "Fantasy Helper (unofficial)", what-ifs in a separate Scenario Lab. IA: Home/This Weekend, Fantasy Helper, Track Record, How it works (+ Scenario Lab, Race Explorer); Model Performance merges into Track Record. Uncertainty: until G3, win/podium shown as bands (no band above 50%, given over-confident favourites), no decimals, DNF only as team retirement rate. Free at launch with email capture; revisit paid (season pass ~£19–29) after 2027 R6; DMCC subscription rules now expected January 2027. Validation: three-GP shadow challenge on the last three 2026 races (22 Nov, 29 Nov, 6 Dec), survey and 8–12 task interviews in Dec–Jan, usability round 2 in Feb pre-season testing. **Hand-over to D/E:** fantasy points must come from the Monte Carlo (finish and qualifying distributions, p10/p90), under a versioned scoring-rules file; today's `fantasy.py` uses a heuristic finish distribution, no qualifying points before quali, out-of-date values (DNF −15 vs −20, fastest lap +5 vs +10, sprint DNF −5 vs −10), and silently defaults all prices to $10M. Owner decisions are listed in MVP_SPEC section 10.2.
