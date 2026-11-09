# EXP-004: Model vs betting market (gate G1)

**Status:** done. **Verdict: the model loses to the market at every stage.** Positioning: fantasy and race-insight tool, not "value bets" (see the Verdict section).
**Evaluation:** the same walk-forward probabilities as `ml/reports/EVAL.md` (each race predicted only from earlier races), scored on the same races as Polymarket's pre-race prices. 50 races for the post-quali winner market (2024 Italian GP to 2026 Mexico City GP).
**Branch:** `claude/market-benchmark-25777b`. No model artifact was written or promoted.

## Plain-language summary

Bettors don't care whether we beat "the pole-sitter wins". They care whether we beat the odds. So we took the betting market's prices before each race, turned them into probabilities, and scored them exactly the way we score our model.

**The market is much better than our model.** After qualifying, the market's average log-loss is 1.17 and ours is 1.49 (the grid baseline is 1.56). The gap is far larger than the noise: the 95% interval for "model minus market" is +0.11 to +0.51, and we beat the market in only 16 of 50 races. Before the weekend it's the same story: 1.99 for the model vs 1.74 for the market.

Even mixing our model 50/50 with the market makes the market slightly *worse* (1.24 vs 1.17). So the model doesn't have useful information that the market lacks. A simulated "bet whenever the model likes a driver more than the market does" strategy after qualifying lost about half the money staked.

## Questions

1. Where can we get historical pre-race F1 odds for 2022–2026 legally, and at which timestamps?
2. On the same races, is the model better than, equal to, or worse than the market (with the margin removed)? At which stage?
3. Is there any sign of a betting edge?

## 1. Odds sources

| Source | Coverage | Timestamp | Access, licence and terms | Used? |
|---|---|---|---|---|
| **Polymarket** (prediction exchange). Gamma API for markets; CLOB `prices-history` for price history | Per-race "Grand Prix Winner" markets from the **2024 Italian GP** onward: 8 races in 2024, 23 in 2025, 20 in 2026 (51 races). "Driver Podium Finish" from the 2025 Brazilian GP onward (21 races). Pole, sprint and constructor head-to-head markets too, but thin. Typical matched volume is about $0.8M per race (median). | Price history at 10-minute resolution, so we can take exact pre-weekend, pre-quali and post-quali snapshots. | Public, documented market-data API with **no account or key** (docs.polymarket.com, "Market Data"). The polymarket.com Terms page could not be fetched from this network, so the ToS has **not** been reviewed. Polymarket blocks trading for UK users (its events are flagged `restricted`). Raw prices are cached in git-ignored `ml/data/cache/market/` and **not committed**. Only aggregate scores are in the repo. | **Yes** |
| **Betfair Exchange historic data** (historicdata.betfair.com) | Free "Basic" tier: last traded price at 1-minute intervals. Betfair lists "other sports" beyond racing, soccer, tennis, cricket and golf. **F1/motor-sport coverage not verified.** | Would allow pre- and post-quali snapshots. | Needs a **Betfair account login**, which is account-gated, so we didn't sign up (per the brief). Download terms are shown at checkout and weren't reviewed. | No. Owner decision needed. |
| Bookmaker odds archives (OddsPortal-type sites) | Some F1 outright history | Usually only the opening or closing price | Terms forbid scraping or automated collection. | No |
| Formula1.com "betting guide" articles (official odds partner) | Selected races, pre-weekend | Single snapshot per article | Editorial pages, no data feed. Reuse not permitted. | No |
| Paid historical-odds APIs (The Odds API, SportsDataIO, OpticOdds, Sportradar) | Bookmaker F1 odds | Varies | Paid | No. Owner decision needed. |
| Kalshi (US prediction exchange) | F1 markets reportedly from 2025 | — | Public API. Not checked in this experiment. | No (possible cross-check) |
| OpenF1 `sessions` | 2023+ session schedule | — | Free, already used by the project | Yes, for session start times only |

There's no source we could use for **2022–2023, or early 2024**. The benchmark therefore covers 2024 R16 to 2026 R19. That's 50 races, overlapping the last 50 of the 91 walk-forward races.

## 2. From prices to probabilities

**How a Polymarket price reads.** Each driver has a "YES" share that pays $1 if he wins. If Verstappen's YES trades at $0.30, the crowd thinks he has roughly a 30% chance. The price we use is Polymarket's displayed price: the midpoint between the best buy and sell offers, or the last trade if that gap is wider than 10¢.

**The overround (the margin).** If you add up every driver's price, you get a bit more than 1.00. Fair probabilities would sum to exactly 100%. The excess is the "overround", the market's built-in edge, the same idea as a bookmaker's margin. In F1 terms, it's like a team ordering more tyres than the race needs: the extra is the cushion, not real demand. Here the books summed to **1.047 on average just before the race** (a 4.7% overround) and 1.07–1.08 earlier in the weekend, when thinly traded longshots are over-priced. UK bookmaker F1 outright books usually carry a bigger margin than this.

**Removing it.** The main method is *proportional*: divide every driver's price by the book total, so the probabilities add to 100% (or 300% for the podium market, since three drivers finish on the podium). As a check we also used the *power* method, which shrinks longshots more (a common correction for the "favourite–longshot bias", where punters overpay for outsiders). The two give practically the same scores (1.172 vs 1.169 post-quali), so the conclusion doesn't depend on the method.

**Details that keep the comparison fair:**
- A driver missing from a market gets an equal share of any "Other" outcome's price. If the market has no "Other" outcome, he gets the 0.1% minimum tick.
- Outcomes naming a driver who isn't in that race (for example a replaced driver) are void. Their stale prices are dropped.
- A book whose total is outside 0.85–1.35 (2.55–4.05 for the podium) is treated as stale or broken and skipped. This removed one pre-weekend winner book (2024 United States GP, book total 1.57), one post-quali podium book and six pre-weekend podium books. The 2026 Brazilian GP (R20) has prices but no result yet, so it isn't scored.

**Snapshots** (scheduled session starts from OpenF1):

| Snapshot | Cutoff | What the market knows | Matching model stage |
|---|---|---|---|
| `pre_weekend` | FP1 start | Same as the model: nothing from this weekend | `pre_quali` (like-for-like) |
| `pre_quali` | Qualifying start | Practice, and on 2024+ sprint weekends the sprint too | `pre_quali` (the market has more information) |
| `post_quali` | 5 min before lights out | Qualifying, the official grid with penalties, the sprint, and the weather on the day | `post_quali` |

## 3. Results

Lower is better for log-loss and Brier. Log-loss is the average "surprise" at the actual winner: giving the winner 30% scores 1.20, giving him 10% scores 2.30. Brier is the mean squared error of every driver's probability. "Hit" is how often the top-rated driver won.

### Winner market, after qualifying (50 races; 14 were sprint weekends)

| Forecast | Log-loss | Brier | Top-pick hit |
|---|---:|---:|---:|
| **Market** (Polymarket, de-margined) | **1.172** | **0.0287** | 52.0% |
| Model (walk-forward post-quali) | 1.485 | 0.0364 | 38.0% |
| Official grid baseline | 1.563 | 0.0307 | 62.0% |
| 50/50 model + market blend | 1.243 | 0.0310 | 48.0% |

- **Model minus market: +0.313** (95% bootstrap CI +0.108 to +0.509). The model is better in 16 of 50 races.
- **Blend minus market: +0.071** (CI −0.029 to +0.167). Adding the model to the market doesn't improve it, and the point estimate is worse.
- Non-sprint weekends only (36 races): model 1.415 vs market 1.063. The gap is not caused by the market knowing sprint results.
- By season: 2024 (8 races) model 1.716 vs market 1.698. **2025 (23) model 1.279 vs market 0.995. 2026 (19) model 1.637 vs market 1.165.**
- On these 50 races the model beats the grid on log-loss (1.485 vs 1.563) but loses on Brier (0.0364 vs 0.0307). The over-confidence row in the next table is why.

**Calibration** asks: when a forecast says X%, does it happen X% of the time?

| Predicted win chance | Model: n / said / happened | Market: n / said / happened |
|---|---|---|
| 0–2% | 778 / 0.2% / 0.1% | 772 / 0.3% / 0.0% |
| 2–5% | 69 / 3.2% / 5.8% | 76 / 3.4% / 2.6% |
| 5–10% | 43 / 7.3% / 7.0% | 48 / 7.0% / 4.2% |
| 10–20% | 55 / 14.4% / 16.4% | 49 / 14.9% / 16.3% |
| 20–35% | 37 / 25.7% / 35.1% | 42 / 27.8% / 33.3% |
| 35–50% | 27 / 42.0% / 40.7% | 19 / 41.1% / 31.6% |
| **50–100%** | **23 / 62.9% / 39.1%** | 26 / 58.8% / 69.2% |

The model's biggest weakness is clear here. When it makes a driver a strong favourite (above 50%), that driver wins only 39% of the time. The market's strong favourites win 69%. Each bucket holds only 20–50 driver-races, so treat single rows as rough.

### Winner market, before the weekend (48 races)

| Forecast | Log-loss | Brier | Top-pick hit |
|---|---:|---:|---:|
| **Market before FP1** | **1.735** | **0.0384** | 33.3% |
| Model (walk-forward pre-quali) | 1.989 | 0.0405 | 27.1% |
| Prior-form baseline | 2.327 | 0.0423 | 25.0% |
| 50/50 blend | 1.784 | 0.0389 | 27.1% |

Model minus market: **+0.254** (CI +0.017 to +0.507). The model is better in 19 of 48 races. 2024 is the only season the model edges it (7 races: 2.074 vs 2.153). Against the market just before qualifying, which has seen practice, the gap widens to +0.333 (model 2.008 vs market 1.675).

### Podium market (only 2025 R23 onward, so mostly 2026)

| Stage | Races | Model log-loss | Market log-loss | Model − market (95% CI) |
|---|---:|---:|---:|---|
| After qualifying | 19 | 0.250 | 0.193 | +0.056 (+0.022 to +0.092). Model better in 4 of 19. |
| Before the weekend | 13 | 0.227 | 0.220 | +0.007 (−0.018 to +0.029). Roughly level, tiny sample. |

Podium log-loss here is the average yes/no log-loss per driver, which is why the numbers are on a different scale from the winner market. Before the weekend the model is roughly level with the market on podiums. But 13 races is far too few to claim anything, and the market's podium books before the weekend are often stale (wide spreads).

### Head-to-head

Polymarket only lists a few constructor head-to-heads with almost no volume (about $1–13k), and there are no driver head-to-heads. Not scored.

### Simulated betting: "back the driver when the model > the market"

Flat 1-unit stake on every driver where the model's probability is above the market's by at least the edge shown. We pay the **raw** price, margin included, and a winning unit returns 1/price.

| Market / stage | Edge | Bets | Winners | ROI | ROI 95% CI | ROI after 5% commission |
|---|---:|---:|---:|---:|---|---:|
| Winner, post-quali | 0% | 160 | 16 | **−51%** | −77% to −17% | −53% |
| Winner, post-quali | 25% | 131 | 9 | −55% | −86% to −11% | −57% |
| Winner, pre-weekend | 0% | 218 | 19 | −4% | −50% to +52% | −8% |
| Winner, pre-weekend | 50% | 144 | 10 | +19% | −51% to +104% | +14% |
| Podium, post-quali | 0% | 69 | 30 | −15% | −38% to +9% | −17% |
| Podium, pre-weekend | 0% | 99 | 18 | −50% | −66% to −30% | −51% |

The full tables for each edge are in `ml/reports/MARKET_BENCHMARK.md`.

**Honest caveats.**
- **Sample size.** 50 races and 100–200 bets, mostly on longshots priced around 5–12¢. One 20-to-1 winner moves ROI by about 15 points. The pre-weekend +19% at a 50% edge comes from 10 winners, and its interval runs from −51% to +104%. That is noise, not an edge, and it sits next to clearly negative post-quali and podium results.
- **The price you'd really get is worse.** We bet at the displayed midpoint. A real buy pays the ask, which is half a spread worse, plus any fee. A UK punter would bet with a bookmaker, at prices with a bigger margin, or on Betfair, paying 2–5% commission on winnings. We show a 5% version.
- **Timing.** Snapshots are the last price before the cutoff, at 10-minute resolution. The post-quali market already knows the morning weather, and on sprint weekends the sprint result. Even on non-sprint weekends the gap stays at 0.35.
- **Polymarket is not the UK market.** UK users can't trade on it. It's a liquid, low-margin crowd forecast, a fair stand-in for "the market". The Betfair Exchange would be the UK benchmark, and we expect it to be at least as sharp. Bookmakers' fair odds are usually close to the exchange's once their larger margin is removed.

## 4. Verdict for gate G1

**The model loses to the market at every stage.**

| Stage | Verdict | Evidence |
|---|---|---|
| After qualifying (winner) | **Loses, clearly** | +0.31 log-loss, CI excludes zero. Better in 16 of 50 races. Strong favourites are badly over-confident. A blend doesn't add value. |
| Before the weekend (winner) | **Loses** | +0.25, CI only just excludes zero. Better in 19 of 48. |
| Podium | Loses after quali, roughly level before the weekend (13 races) | Too little data to claim anything before the weekend. |

What this means for positioning (the roadmap's own rule):
- **Don't sell "value bets" or "beat the bookie" claims.** In 50 races, the model's disagreements with the market were mostly the model being wrong. Making value claims to UK/EU bettors with this evidence would be misleading, and in the UK it is a gambling-advertising and consumer-protection risk.
- **Position for G2 as a fantasy and race-insight tool.** Explain *why* drivers are fancied (form, track type, grid importance, reliability), cover scenarios and what-ifs, and give fantasy value. Our pre-weekend forecast is a reasonable starting view (it beats the prior-form baseline by about 0.34), but it isn't sharper than the market.
- If bettors stay a target later, show the **market's de-margined probabilities next to ours**, framed as information rather than tips. Re-run this benchmark every race, and only revisit "value" positioning if the live record beats the market over a full season.

## 5. What to try next (for workstream D)

1. **Calibrate the favourites.** The 50%+ bucket says 63% and happens 39%. Fit the calibration to post-quali favourites, for example a separate temperature for the top of the grid, and re-score against the market.
2. **Use the market as a feature, or as the baseline to beat.** For a product, a "market + model adjustments" forecast is the realistic route to something sharper than either alone. The 50/50 blend already beats the model by a wide margin (1.24 vs 1.49), though not the market.
3. **Ask the owner about a Betfair account** to pull Basic historic data. Check F1 coverage first, then repeat the benchmark against the UK exchange and extend it back to 2022.
4. Keep saving market snapshots alongside every live forecast (`race_weekend.py`), so the 2027 track record can be published against the market.

## Reproduce

```bash
cd ml
python scripts/market_benchmark.py all      # fetch prices + session times, walk-forward predictions, report (several minutes; the walk-forward refits every fold)
python scripts/market_benchmark.py report   # re-score from the cached files only
```

- Prices are fetched into `ml/data/cache/market/` (git-ignored, not committed).
- Walk-forward probabilities come from `f1ml.train.train_and_eval(write_reports=False, predictions_out=...)`. That's the same folds and code as `EVAL.md`, starting at the 2024 season (`F1ML_EVAL_FROM_SEASON=2024` only skips earlier test races; training is unchanged).
- Outputs: `ml/reports/MARKET_BENCHMARK.md` (full tables) and `ml/reports/market_benchmark.json`.
- The numbers above come from the run on 2026-11-09, after the evaluation-field fix in `docs/ML_LAUNCH_READINESS.md` (item 7).

## Limitations

- No usable odds for 2022–2023. The benchmark covers 50 of the 91 walk-forward races, with only 8 from 2024.
- One market (Polymarket) and one price type (displayed mid). Bookmaker and Betfair prices were not tested.
- Session cutoffs use scheduled start times. Delayed sessions aren't adjusted for.
- Podium coverage starts at the end of 2025. No usable head-to-head markets.
