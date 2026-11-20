# Pitstop Oracle — Product Requirements (v3)

> **Source:** converted from `docs/v2/Pitstop_Oracle_Product_PRD_v3.pdf` on 2026-11-09. The content below is faithful to the PDF.
>
> **Status vs current code (coordinator note, 2026-11-09):**
> - The PRD was written from the public README, not from a source audit.
> - Some of its open questions are now answered:
>   - Walk-forward evaluation is separate for pre-quali and post-quali (`ml/reports/EVAL.md`).
>   - Outputs are Monte Carlo probabilities, with calibration measured. The model is over-confident on strong favourites (EXP-004).
>   - The model now *does* edge the pole/grid baseline on log-loss overall, but not in 2026 and not on Brier score.
> - **Product direction changed after this PRD:** we are now explicitly an F1 analytics and **fantasy helper** with no betting features ([`ROADMAP_2027.md`](ROADMAP_2027.md)). That conflicts with this PRD's non-goal "no claim that Fantasy Lab is an official F1 fantasy optimizer", and with renaming Fantasy Lab to Scenario Lab. Workstream C must resolve this. Likely answer: an *unofficial* fantasy helper, clearly labelled, with Scenario Lab as a separate page.

## 1. Executive summary
Pitstop Oracle is a fan-facing Formula 1 forecasting and exploration product. Its current Streamlit application has four main experiences: This Weekend, Race Explorer, Fantasy Lab and Model Performance.

**Direction:** refine the existing application rather than rebuild it. Help fans understand the next Grand Prix, inspect a forecast, explore historical outcomes and test scenarios.

The public README reports that the current ML model does not beat the pole-position baseline on its test set. Treat this as a transparency feature and a product-learning opportunity, not something to hide.

## 2. Current product baseline
| Page | README-described behavior | Role |
|---|---|---|
| This Weekend | Selects the next GP and predicts, including before qualifying | Primary landing page |
| Race Explorer | Historical races; model prediction versus actual winner | Evidence and exploration |
| Fantasy Lab | Wet-race and grid-penalty what-if toggles | Scenario experience |
| Model Performance | Reports model performance against simple baselines | Trust and transparency |

README-level pipeline: Jolpica API and Open-Meteo → `ingest.py` → raw files → `features.py` → `driver_race.parquet` → `train.py` → joblib models → Streamlit UI. The README describes `dataCollectionService` as a future/optional Postgres warehouse.

## 3. Users and jobs to be done
- **Primary audience:** engaged F1 fans who follow race weekends and want context around contenders and uncertainty.
- **Secondary audience:** data-curious fans, creators, newsletter writers and publishers.

Jobs:
- Before a race: understand the likely contenders and how uncertain the forecast is.
- After qualifying: understand how the grid changes the outlook.
- Explore historical races, and compare the model with actual outcomes and simple baselines.
- Test understandable hypothetical scenarios, such as rain or grid penalties.
- Over a season: review the forecast track record.

## 4. Goals and non-goals
**Goals**
- Make the next-race forecast easy to understand and revisit.
- Show the event, timestamp, forecast stage, data freshness and model version.
- Compare fairly with the pole sitter and other simple baselines.
- Explain scenario assumptions and limitations.
- Validate demand before building accounts, leagues or payments.

**Non-goals**
- No full rewrite.
- No claim that ML beats the pole baseline.
- No telemetry or paid-data dependency by default.
- No premature user-account or league system.
- No claim that Fantasy Lab is an official F1 fantasy optimizer.

## 5. Product requirements
- **This Weekend (P0):** show the Grand Prix, date, forecast stage, contender ranking, generated-at time, data-as-of time, model version and data warnings. Display probabilities only if the outputs are calibrated probabilities. Otherwise call them scores or rankings.
- **Race Explorer (P1):** preserve historical browsing and the model-versus-actual comparison. Distinguish historical replay from forecasts generated with the information available at the time.
- **Scenario Lab (P1; currently Fantasy Lab):** keep the wet-race and grid-penalty controls only if their assumptions are implemented and testable. Show baseline versus scenario, and label the result hypothetical. Consider renaming to Scenario Lab.
- **Model Performance (P0):** show the sample size, date range, target definition, split and baselines. Separate pre-qualifying from post-qualifying evaluation, and show model versions.

## 6. Success metrics and validation
| Metric | Measurement |
|---|---|
| Activation | Visitors opening This Weekend and inspecting a contender |
| Repeat use | Returning visitors across consecutive GPs |
| Scenario engagement | Scenario usage and completion |
| Trust | User comprehension of forecast limitations |
| Prediction-game demand | Manual challenge submissions and repeat participation |

Validation plan:
- Recruit 30–50 survey respondents across casual, regular and data-oriented fans.
- Conduct 8–12 interviews.
- Test the current app through tasks, not only opinion questions.
- Run a manual prediction challenge over three GPs before building a social layer.

## 7. Delivery sequence and risks
- **Phase 0 — Reliability:** event selection, data freshness, forecast-stage labels and error states.
- **Phase 1 — Trust:** baseline comparison, metric definitions, model versioning and forecast history.
- **Phase 2 — Experience:** improve the existing four pages.
- **Phase 3 — Demand:** run a manual three-GP prediction challenge.
- **Phase 4 — Expansion:** add accounts, history, leagues or publisher tools only when validated.

**Open questions**
- Are the outputs calibrated probabilities or raw scores?
- Can historical data be reconstructed as of the forecast cutoff?
- Are the pre- and post-qualifying evaluations separate?
- Do the scenario toggles reflect a model or a heuristic?

**Scope limitation:** this PRD is based on the public README and repository overview, not a complete source-code audit. Verify implementation details in `ingest.py`, `features.py`, `train.py`, `predict.py`, `ui.py` and `metrics.json`.
