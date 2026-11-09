# Pitstop Oracle: market research (workstream A)

**Author:** workstream A session (Claude), for Siddharth Bisht.
**Date:** 2026-11-09. **Status:** desk research done; nothing here has been validated with real users yet.
**Read with:** [`ROADMAP_2027.md`](ROADMAP_2027.md). The final positioning choice waits for gate **G1** (the model-vs-market benchmark, workstream B).

> **Not legal advice.** Section 3 summarises public guidance so we know what questions to ask. Before launch, a UK gambling/advertising lawyer must review anything marked ⚖️.

Every factual claim links to its source. Where a source is weak (a vendor blog, a third-party price tracker, an undated page), the text says so. Words in *italics* on first use are in the [glossary](#8-glossary).

---

## Executive summary

1. **Two audiences, very different sizes and wallets.**
   - **F1 Fantasy players** are the bigger, easier audience. The official game had a record [2.75 million+ teams in 2025](https://www.fanamp.com/articles/mini-leagues-thrive-in-another-successful-season-for-f1-fantasy-2025), up from [1.5 million sign-ups in 2024](https://www.grandprix247.com/formula-1-news/join-the-gp247-f1-fantasy-league-this-season). They pay little: the leading tool, F1 Fantasy Tools, charges [€2.50–€6 a month (billed yearly)](https://f1fantasytools.com/pricing), with a generous free tier.
   - **UK/EU F1 bettors** pay more, but F1 is a small betting sport: about [0.4% of global betting turnover](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/), according to F1's own commercial director. Paid tipsters and value-betting tools charge [£20–£40 a month](https://bmsmarketing.thrivecart.com/bt---quickfire-winners) and up to [~€89+ a month](https://hiltonsoftware.pages.dev/compare/rebelbetting-vs-trademate-sports) for serious tools.
2. **The F1 prediction space is crowded with free, unproven models.** Sites such as [f1predictions.ai](https://f1predictions.ai/predictions), [F1 Predictor AI](https://f1predictor.me/) and [WillWin](https://willwin.ai/f1/) already publish free win probabilities. None that we found publishes a proper probability-quality score (log-loss, calibration), or compares itself with bookmaker odds. **An honest, proofed track record is the gap we can own.**
3. **Odds tools are going free.** Oddschecker shut its paid OC+ subscription (formerly [£29.99–£34.99 a month](https://www.oddschecker.com/insight/specials/20251127-oddschecker-black-friday-special-get-over-50pp-off-ocplus-premium-and-a-7-day-free-trial)) and made all its tools, including "AI probabilities", [free from 28 October 2026](https://www.oddschecker.com/tips/horse-racing/20261028-oddscheckerplus-is-now-free-unlimited-odds-comparison-expert-betting-tips-return). F1 itself now publishes free, age-gated betting content with official operators [Betway (Europe, Middle East and Africa)](https://www.blackbookmotorsport.com/news/f1-betway-betting-operator-march-2026/) and [FanDuel (North America)](https://www.covers.com/industry/fanduel-official-betting-operator-f1-north-america-april-30-2026). Generic "odds plus AI probabilities" will be hard to charge for.
4. **Regulation (UK).** Selling advice only, with no bets handled, appears not to need a Gambling Commission licence. A tipster who *places bets for customers* does need one ([Gambling Commission guidance](https://www.gamblingcommission.gov.uk/licensees-and-businesses/guide/betting-advice-for-remote-non-remote-and-betting-intermediaries); [SBC News, 2015](https://sbcnews.co.uk/europe/uk/2015/10/15/betting-gods-darren-moore-should-sports-tipsters-be-regulated/)). ⚖️ Confirm this. The real constraints are the **ASA/CAP advertising rules**: track-record claims must be *proofed*, and we must not imply that a tip is likely to win ([ASA tipster guidance](https://asa.org.uk/advice-online/betting-and-gaming-tipsters.html)). Bookmaker affiliate links bring in the gambling-ad rules, including the [ban on content of "strong appeal" to under-18s](https://www.asa.org.uk/news/cap-and-bcap-update-guidance-on-protecting-under-18s-in-gambling-and-lotteries-advertising.html). New UK **subscription rules** (renewal reminders, easy cancellation) are due to apply from [1 January 2027](https://www.winstontaylor.com/insights/u-k-s-subscription-contract-rules-take-effect-on-january-1-2027-draft-legislation-and-guidance-published), about ten weeks before our launch.
5. **The EU is a patchwork.** Italy bans gambling advertising, including affiliates ([Mondaq](https://www.mondaq.com:443/italy/gaming/741432/ban-on-gambling-advertising-in-italy)). Belgium banned most gambling ads and raised the gambling age to 21 ([Belga](https://www.belganewsagency.eu/belgium-bans-gambling-advertisements); [iGB](https://igamingbusiness.com/legal-compliance/belgium-implements-new-rules-banning-all-under-21s-from-gambling/)). The Netherlands bans untargeted ads and is [considering a full ban](https://www.yogonet.com/international/news/2026/06/16/123982-netherlands-proposes-gambling-advertising-ban-with-deposit-limits-and-affordability-checks). Germany requires [fixed, not revenue-linked, affiliate pay](https://cdn.qualityunit.com/blog/glustv-2021-affiliate-marketing-compliance). A fantasy and analytics product travels across the EU far more easily than a betting-tips product.
6. **Recommendation: Option B, "transparent F1 analytics plus a fantasy helper"**, with a public, timestamped track record from day one, and a free *market-check* view that shows our probabilities next to bookmaker odds, with no "bet now" language. Upgrade to Option A ("value-odds model") only if G1 shows that our probabilities are at least as accurate as the *de-vigged* market. Today's model only just beats the starting grid ([`ROADMAP_2027.md`](ROADMAP_2027.md)), so Option A is not supportable yet. Option C ("tipster") needs years of profitable proofed results that F1's ~24 races a year can't deliver before launch.

---

## What we sell today (context)
From [`README.md`](../../README.md), [`docs/README.md`](../README.md), `ml/pages/` and `ml/f1ml/fantasy.py`:

| Today | What it does |
|---|---|
| **This Weekend** (`1_This_Weekend.py`) | Win probabilities for the next GP, before and after qualifying. |
| **Race Explorer** (`2_Race_Explorer.py`) | Past races: model pick vs pole vs actual winner. |
| **Fantasy Lab** (`3_Fantasy_Lab.py`) | What-ifs (rain, grid penalties). `fantasy.py` adds F1-Fantasy-style expected points and a budget-constrained team optimizer. |
| **Model Performance** (`4_Model_Performance.py`) | Honest accuracy vs the pole baseline. |

Not built yet: odds ingestion, market comparison, accounts, payments, a public track-record page.

---

## 1. Segments

### 1.1 Summary table

| Segment | Rough size (best available evidence) | What they use now | Pain points | What they already pay for |
|---|---|---|---|---|
| **Casual UK/EU F1 bettor**: a few £ on the winner or podium, mostly for fun | 8.4% of GB adults bet online on sports or racing in a typical four weeks ([GSGB 2025, secondary write-up](https://sigma.world/news/half-adults-britain-gamble-survey-2025/)). F1 is ~0.4% of global betting turnover ([BlackBook](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/)). At Entain, the UK is the biggest source of F1 bettors (35%) ([Gambling Insider](https://www.gamblinginsider.com/news/18634/entain-reports-rise-in-formula-1-wagers)). No reliable count of UK F1 bettors exists. | Bookmaker apps, free tips (Sporting Life's [Weekend Best Bets podcast](https://www.sportinglife.com/racing/news/podcast-weekend-best-bets/170838)), [F1's own betting guide](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/) | Few F1 markets (champion, race winner; [in-play markets still being built](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/)). Hard to judge whether a price is fair. Tips are opinions without a record. | Usually nothing. Free content and bookmaker offers. |
| **Value-seeking bettor**: compares prices, wants an edge | Unknown share. The market for paid value tools proves the type exists ([RebelBetting, Trademate, OddsJam](https://hiltonsoftware.pages.dev/compare/rebelbetting-vs-trademate-sports)). | Odds comparison ([Oddschecker, now free](https://www.oddschecker.com/tips/horse-racing/20261028-oddscheckerplus-is-now-free-unlimited-odds-comparison-expert-betting-tips-return)), value-bet scanners, tipster vetting ([Smart Betting Club](https://www.gambling.com/uk/news/should-you-pay-for-tips-the-knowledge-ep1-pete-ling-of-sbc-2091000)) | Value tools target football, tennis and racing. None we found models F1 from first principles. Unverified tipster claims ([ASA rulings](https://www.asa.org.uk/rulings/serious-profits-syndicate-a12-190288.html)). | Value software (~€89+/mo, unverified), tipsters (£10–£40/mo), SBC (~$139.99/yr, from an affiliate page) ([source](https://bettoredge.com/post/tipster-platforms-comparison)) |
| **Sharp bettor**: models their own prices, bets into the sharpest markets | Small. No public data. | Their own models, sharp bookmakers and exchanges | Bookmakers limit winning accounts (widely reported, but not sourced here). Thin F1 markets. | Data, not opinions. They would pay only for raw probabilities that demonstrably beat the market. |
| **Betfair exchange trader**: trades price swings, often in-play | No F1-specific matched-volume data. Volume figures online are unsourced ([example of a weak source](https://sparkco.ai/blog/formula-1-championship-prediction-markets)). | Betfair Exchange (offers [both a sportsbook and an exchange for F1](https://www.squawka.com/en/?p=1471060)). Trading software. | Thin liquidity in niche markets makes spreads wider ([Whelan, academic paper on Betfair](https://www.karlwhelan.com/Papers/Betfair.pdf)). A liquid exchange market's *overround* is around 102% ([Sporting Life](https://www.sportinglife.com/free-bets/guides/betfair-exchange-education/favourites-and-longshots)). | Exchange commission, trading tools. Would want live, in-race probabilities, which we don't have. |
| **Official F1 Fantasy player** | **2.75M+ teams in 2025** (a record; teams, not people, and up to 3 teams per account in 2026) ([FanAmp](https://www.fanamp.com/articles/mini-leagues-thrive-in-another-successful-season-for-f1-fantasy-2025); [Motorsport.com](https://www.motorsport.com/f1/news/f1-fantasy-2026-prizes-and-changes-coming-to-this-season-explained/10800060/)) | F1 Fantasy app, FanAmp articles ([official content partner](https://www.fanamp.com/articles/f1-fantasy-tips-to-start-season-strong-2026)), F1 Fantasy Tools, Reddit | Transfer and chip timing. Price-change prediction ([Budget Builder](https://www.fanamp.com/f1-news/introducing-league-analyzer-from-f1-fantasy-tools)). "Who will score points this weekend?" | Small subscriptions: **€2.50–€6/mo, billed yearly** ([F1 Fantasy Tools](https://f1fantasytools.com/pricing)) |
| **Other fantasy players** (GridRival, FanTeam and similar) | Small. FanTeam's 2025 season game had a **£10,000 guaranteed prize pool** ([FISO](https://www.fiso.co.uk/fanteams-formula-fantasy-2025-season-game/)). GridRival is free-play outside the US ([The Race, promoted article](https://www.the-race.com/promoted/fantasy-f1-game-showdown-gridrival-vs-fantasy-f1/)). | Game apps, FISO-style forums | Paid-entry players care about winning money, so they behave closer to bettors | Entry fees |

**Plain-English takeaway.** Think of fantasy players as the big midfield pack: lots of them, each paying a little. Bettors are the front-runners: fewer, more willing to pay, but extremely demanding, and they compare us with the betting market, not with the starting grid.

### 1.2 Notes per segment
- **The F1 audience skews young.** The 2025 Global Fan Survey (100,000+ responses) highlights Gen Z, women and US fans as growth groups ([Formula1.com](https://www.formula1.com/en/latest/article/formula-1-and-motorsport-network-unveil-2025-global-fan-survey.4YqMebNy8BLaapyJfjzDXO); [BlackBook](https://www.blackbookmotorsport.com/news/f1-motorsport-network-study-women-gen-z-fans-sponsors-july-2025/)). 58% of motorsport bettors are aged 18–34 ([BlackBook](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/)). A young audience helps fantasy, but creates a real risk under the under-18 advertising rules (section 3.3).
- **F1 says its fans are more inclined to bet than fans of other sports**, and it targets "avid fans who bet elsewhere" ([BlackBook](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/)). That makes F1 a competitor for the casual bettor's attention.
- **Online sports betting in GB is male-skewed**: 13.3% of men vs 3.8% of women in four weeks ([Sigma, citing GSGB tables](https://sigma.world/news/half-adults-britain-gamble-survey-2025/)). ⚠️ Secondary source; check against the [official GSGB report](https://www.gamblingcommission.gov.uk/statistics-and-research/publication/gambling-survey-for-great-britain-annual-report-2025-official-statistics).

---

## 2. Competitors

### 2.1 Main table

| Competitor | Type | Key features | Pricing | Positioning | Publishes a track record? | Weakness we could exploit |
|---|---|---|---|---|---|---|
| [f1predictions.ai](https://f1predictions.ai/predictions) | F1 AI predictions | Winner, podium, pole, sprint, head-to-heads, season forecast, "Beat the AI" game | Free, plus an "Unlock Pro" tier (price not shown) | "AI predictions, scored in public" | **Yes**: an accuracy tracker (52% average score, 16/23 races) | Scores hits and misses, not probability quality. No market comparison. Our closest competitor. |
| [F1 Predictor AI](https://f1predictor.me/) | F1 simulator | Load the qualifying order and get win, podium and top-10 probabilities. Chat feature powered by Claude. | Free | Hobby tool by one developer | Weak: a 5-race 2024 validation. Unexplained "98.3% ROC-AUC" claim. | Tiny validation. No weather or DNF modelling (its own stated limits). |
| [WillWin](https://willwin.ai/f1/) | LLM championship forecast | A nightly language-model title probability per driver | Free | Experiment. Says itself that it is "frequently wrong". | No | Championship only, not race by race |
| [ThePredictX](https://thepredictx.com/sports/formula1) | Prediction-market style | Yes/no probabilities on red flags, safety cars, rain | Not checked | Live prop-style probabilities | Not checked | Different product (event props) |
| [F1 Fantasy Tools](https://f1fantasytools.com/pricing) | Fantasy tool suite | Team Calculator, Budget Builder (price changes), live scoring, weekly simulations, league and rival analysis | Free tier. **€2.50 / €4.50 / €6 a month, billed yearly**. Pro also monthly on Patreon. | The fantasy player's toolbox | No race-outcome track record | Their simulations rest on price and points data, not a published race-outcome model. **Partner or differentiate, not fight head-on.** |
| [FanAmp](https://www.fanamp.com/articles/f1-fantasy-tips-to-start-season-strong-2026) | Fantasy content | Tips, guides, community surveys | Free content | Official-looking fantasy media | No | Editorial opinion, not quantified |
| [GridRival](https://the-race.com/formula-1/gridrival-adds-daily-fantasy-sports-games) / [FanTeam](https://www.fiso.co.uk/fanteams-formula-fantasy-2025-season-game/) | Alternative fantasy games | Season and daily contests; real money in some regions | Free play / paid entry | Games, not tools | n/a | Their players need projections, which our optimizer could supply |
| [Oddschecker](https://www.oddschecker.com/tips/horse-racing/20261028-oddscheckerplus-is-now-free-unlimited-odds-comparison-expert-betting-tips-return) | Odds comparison and tips | Odds from 25+ bookmakers, price history, value bets, "AI probabilities", expert tips | **Free since 28 Oct 2026** (was £29.99–£34.99/mo) | The UK default for checking prices. Earns from bookmaker affiliate deals. | Not for its AI probabilities (not seen) | Generic across all sports. No F1-specific depth or proven F1 model. |
| [F1 Betting Guide (formula1.com)](https://www.covers.com/industry/fanduel-official-betting-operator-f1-north-america-april-30-2026) | Official editorial | Age-gated betting content with official operators ([Betway](https://www.blackbookmotorsport.com/news/f1-betway-betting-operator-march-2026/), FanDuel); official data from [ALT Sports Data](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/) | Free | The official voice; promotes its partner bookmakers | No | Conflicted (paid by bookmakers). Not independent analysis. |
| [Sporting Life Best Bets](https://www.sportinglife.com/racing/news/podcast-weekend-best-bets/170838), F1 betting podcasts ([F1 Money Gang](https://www.podengine.ai/podcasts/f1-money-gang-podcast), [Apex Insiders](https://apexinsiders.podbean.com/page/4/)) | Free tips | Weekly previews | Free (affiliate-funded) | Pundit opinion | Rarely proofed | Opinion. Several F1 betting podcasts have gone inactive ([F1 Gambling Podcast](https://podcasts.apple.com/podcast/id1638858422)). |
| Generic paid tipsters, e.g. [TipMan](https://www.tipmantips.com/premium-sports-betting-tips/), [QuickFire Winners](https://bmsmarketing.thrivecart.com/bt---quickfire-winners), [Beat the Bookmakers](https://podosfairo.gumroad.com/l/beat-the-bookmakers) | Tipster | Daily tips, mainly football and racing | £169.99–£299.88/yr; £39/mo; €19.90/mo | "Profit" | Self-reported marketing figures | No F1 specialists found. The sector has a trust problem: [repeated ASA rulings](https://www.asa.org.uk/rulings/serious-profits-syndicate-a12-190288.html), and [affiliate tipsters earning ~30% of followers' losses](https://www.irishtimes.com/sport/other-sports/how-online-tipsters-make-a-profit-when-punters-lose-1.3921229). |
| [Smart Betting Club](https://www.gambling.com/uk/news/should-you-pay-for-tips-the-knowledge-ep1-pete-ling-of-sbc-2091000) | Tipster vetting | Independent tracking and reviews of tipsters (founded 2006) | ~$139.99/yr ([affiliate page](https://bettoredge.com/post/tipster-platforms-comparison), unverified) | Trusted referee | n/a | A possible *validator* of our record, not a competitor |
| RebelBetting / Trademate / OddsJam ([comparison, unverified](https://hiltonsoftware.pages.dev/compare/rebelbetting-vs-trademate-sports)) | Value-bet software (non-F1 analogue) | Scans bookmakers for prices above "true" odds | ~€89/mo; ~$49/mo with free plans | Professional edge tools | Varies | F1 is barely covered. They rely on sharp-bookmaker prices, not their own sport models. |
| Action Network PRO / Dimers Pro ([price tracker](https://apppricinglab.com/app/apple/1083677479); [Dimers](https://aigearbase.com/tool/dimers)) | Model-driven picks (US analogue) | Projections, best odds, picks | ~$24.99/mo each (third-party figures) | Data-driven betting app | Partly | US sports focus. Shows the price ceiling for a model-plus-picks app. |
| [Infogol (now part of Timeform)](https://timeform.com/football) | Model-driven probabilities (football analogue) | xG-based home/draw/away probabilities | Free tips page | "Transparent model numbers" | Not checked | **The best analogue for Option B**: a free model probability page that builds trust and funds itself through other products |

### 2.2 What the table tells us
- **Free model probabilities are already common in F1**, so "we have an AI model" is not a selling point.
- **Hardly anyone measures probability quality.** f1predictions.ai scores hits; F1 Predictor AI quotes small samples. Publishing log-loss, calibration and a *market comparison* every weekend would be unusual, and hard to copy quickly, because it needs a timestamped history.
- **Fantasy tools are cheap and good.** We should pitch our fantasy features as *race-outcome intelligence* that F1 Fantasy Tools lacks (who scores points, DNF risk, wet-race what-ifs), not as another optimizer.
- **Betting tools are being commoditised** (Oddschecker going free), and their money comes from bookmaker affiliate deals.

---

## 3. Regulation (UK first, then key EU markets)

> ⚖️ **This is desk research, not legal advice.** Items marked ⚖️ must go to a UK gambling and advertising lawyer before launch.

### 3.1 Do we need a UK Gambling Commission licence?
- **Probably not, if we only sell information.** The Commission licenses betting operators and *betting intermediaries*. Its guidance says a tipster who **places bets on behalf of others** for payment or commission would count as a betting intermediary under section 13 of the Gambling Act 2005, and would need a licence ([Gambling Commission](https://www.gamblingcommission.gov.uk/licensees-and-businesses/guide/betting-advice-for-remote-non-remote-and-betting-intermediaries)). A 2015 report quotes the Commission's position that "if a business is not handling bets then no licence is required" ([SBC News](https://sbcnews.co.uk/europe/uk/2015/10/15/betting-gods-darren-moore-should-sports-tipsters-be-regulated/)).
- **The Commission doesn't license affiliates or tipsters directly.** Instead, it holds the bookmakers responsible for them ([Vice](https://develop.vice.com/en/article/how-twitter-tipsters-are-profiting-on-losing-bets/); [LCCP SR 1.1.2](https://www.gamblingcommission.gov.uk/licensees-and-businesses/lccp/condition/1-1-2-responsibility-for-third-parties-all-licences)).
- ⚖️ **Confirm in writing** that a paid prediction subscription, with no stake handling, no pooled funds, no "we bet for you" and no prize competitions, needs no licence. Also check any future feature that looks like gambling: paid-entry prediction contests with cash prizes could become betting or a prize competition.

### 3.2 Advertising: CAP and BCAP codes (enforced by the ASA)
CAP covers non-broadcast ads, such as websites and social media; BCAP covers TV and radio. Both have specific guidance for betting tipsters:
- **Proof track-record claims.** Claims about past winners or profit should be *proofed*: lodged with an independent third party before the event ([ASA tipster advice](https://asa.org.uk/advice-online/betting-and-gaming-tipsters.html); [BCAP tipster rules](https://www.asa.org.uk/static/bd344c4e-5832-4d91-8bcfdb48c544c00e/0187613b-aa16-4281-ad0f8dae9c9e8a78/The-BCAP-Code-Betting-tipsters.pdf)).
- **Base profit claims on starting price**, or on prices you can prove were available ([ASA ruling, Isiris Racing](https://asa.org.uk/rulings/Isiris-Racing-Services-A13-251637.html)).
- **Don't imply that a tip is likely or certain to win**, or that the service will very likely make a profit. Guarantees have been ruled misleading ([Philip Murray ruling](https://www.asa.org.uk/rulings/philip-murray-a18-455610.html)). Short winning streaks don't support future-profit claims ([Isiris ruling](https://www.asa.org.uk/rulings/Isiris-Racing-Services-A11-169831.html)).
- **Have evidence ready.** Failing to answer the ASA gets a complaint upheld ([Private Investors](https://www.asa.org.uk/rulings/private-investors-a13-226383.html)).

**What this means for us:** our planned live shadow test (roadmap workstream "—") should save each forecast with a tamper-evident timestamp before each race. For example, we could commit a hash of the forecast to a public git repository, or email it to an independent party. ⚖️ Ask whether a public git commit counts as "independent proofing", or whether we need a third party such as a solicitor or Smart Betting Club.

### 3.3 Affiliate links to bookmakers
- If we earn commission from bookmakers, **our pages become gambling advertising**. The bookmaker is held at least jointly responsible for affiliate content under the CAP Code ([ASA](https://www.asa.org.uk/news/cap-and-bcap-update-guidance-on-protecting-under-18s-in-gambling-and-lotteries-advertising.html); [Lewis Silkin](https://www.lewissilkin.com/insights/2022/10/06/updated-rules-banning-gambling-ads-with-strong-appeal-to-children-now-in-force-102hydu)). Bookmakers must also contractually bind affiliates to the licence conditions ([LCCP SR 1.1.2](https://www.gamblingcommission.gov.uk/licensees-and-businesses/lccp/condition/1-1-2-responsibility-for-third-parties-all-licences)). In practice, bookmakers will impose compliance terms on us.
- **"Strong appeal" to under-18s is banned** (CAP rule 16.3.12, since October 2022) ([CAP Code section 16](https://www.asa.org.uk/type/non_broadcast/code_section/16.html)). A rule of thumb: personalities with 100,000+ under-18 followers indicate strong appeal ([iGaming Expert](https://igamingexpert.com/news/affiliates/asa-clarity-high-bar-influencer-follower-threshold/)). **F1 drivers are hugely popular with teenagers**, so driver imagery next to bookmaker links is risky. ⚖️
- **Revenue share vs CPA.** The industry has been criticised for tipsters earning ~30% of followers' losses ([Irish Times](https://www.irishtimes.com/sport/other-sports/how-online-tipsters-make-a-profit-when-punters-lose-1.3921229); [Guardian via inkl](https://www.inkl.com/news/pressure-grows-on-gambling-industry-to-stop-tipsters-profiting-from-losses)). If we use affiliates, a flat fee per sign-up is safer for trust. In Germany it is required: affiliate pay must be fixed, not linked to deposits or losses ([QualityUnit, vendor blog](https://cdn.qualityunit.com/blog/glustv-2021-affiliate-marketing-compliance)).
- **Recommendation:** no affiliate links at launch. Sell subscriptions. This keeps us outside most gambling-advertising rules and protects the "independent" message. Revisit after launch with legal advice.

### 3.4 Age-gating and responsible gambling (RG)
- F1's own betting content is **age-gated** ([BlackBook](https://www.blackbookmotorsport.com/features/f1-betting-jonny-haworth-mark-wrigley-interview/)), and its FanDuel deal includes responsible-gaming content ([Covers](https://www.covers.com/industry/fanduel-official-betting-operator-f1-north-america-april-30-2026)). That sets the bar users and partners will expect.
- **Proposed** (not yet a legal requirement for an advice-only site; ⚖️): put any betting-framed features (odds, market comparison) behind an 18+ confirmation (21+ for Belgian users, [iGB](https://igamingbusiness.com/legal-compliance/belgium-implements-new-rules-banning-all-under-21s-from-gambling/)). Add RG messaging and links (GamCare, BeGambleAware) and a plain "probabilities, not guarantees" note. Keep the fantasy and analytics areas open to all ages.
- The Commission says CAP principles should apply to social media content even where the code isn't explicit ([Lewis Silkin](https://www.lewissilkin.com/insights/2022/10/06/updated-rules-banning-gambling-ads-with-strong-appeal-to-children-now-in-force-102hydu)).

### 3.5 Consumer protection for paid subscriptions
- **The DMCC Act 2024 subscription regime** is expected from **1 January 2027** ([Winston Taylor](https://www.winstontaylor.com/insights/u-k-s-subscription-contract-rules-take-effect-on-january-1-2027-draft-legislation-and-guidance-published); [Fieldfisher](https://www.fieldfisher.com/en/insights/subscription-reforms-what-businesses-need-to-do-now)). It requires clear pre-contract information, renewal reminders, easy cancellation, and a cooling-off window after a trial or a 12-month-plus renewal ([Lewis Silkin](https://www.lewissilkin.com/insights/2026/04/02/consumer-law-update-subscriptions-regime-delayed-again-to-spring-2027-102mops)). Commentators note that the secondary legislation and guidance were still pending ([SubTracker](https://subtracker.io/best/uk-dmcc-subscription-rules)). ⚖️ Our payments setup must comply from day one. Using a hosted billing provider with reminder and cancel flows lowers the effort.
- **Don't promise profit.** Tipsters have been criticised for no-refund policies alongside large profit claims ([TipMan terms](https://www.tipmantips.com/premium-sports-betting-tips/)). The DMCC Act also strengthened the CMA's powers against unfair practices ([EM Law](https://emlaw.co.uk/digital-markets-competition-and-consumers-act-2024-consumer-law-changes-and-enhanced-rights/)).

### 3.6 Key EU differences

| Market | What matters for us | Source |
|---|---|---|
| **Ireland** | New national regulator under the Gambling Regulation Act 2024; provisions phased in from 2026. Advertising rules are still being set. ⚖️ | [iGB](https://igamingbusiness.com/?p=330176); [European Gaming](https://europeangaming.eu/portal/gambling-advertising/) |
| **Italy** | General ban on gambling advertising since 2018/19 (the Dignity Decree), which leaves affiliates little room | [Mondaq](https://www.mondaq.com:443/italy/gaming/741432/ban-on-gambling-advertising-in-italy); [CDC Gaming](https://cdcgaming.com/commentary/the-unbridgeable-gap-of-the-italian-gambling-advertising-and-sponsorship-ban/) |
| **Spain** | Royal Decree 958/2020 limits gambling ads (time windows, no celebrity endorsement, bonus limits) | [IAGR](https://iagr.org/?p=1375); [IMGL](https://www.imgl.org/gaming-regulation/advertising/) |
| **Germany** | GlüStV 2021. Advertising unlicensed operators is banned; courts let the regulator hold operators liable for affiliates; affiliate pay must be fixed. | [iGB](https://igamingbusiness.com/legal-compliance/court-confirms-ggl-affiliate-violations/); [Focus GN](https://focusgn.com/court-agrees-german-regulator-can-hold-operators-responsible-for-affiliates) |
| **Netherlands** | Untargeted gambling ads banned since July 2023; a June 2026 proposal would ban gambling ads entirely | [European Gaming](https://europeangaming.eu/portal/gambling-advertising/); [Yogonet](https://www.yogonet.com/international/news/2026/06/16/123982-netherlands-proposes-gambling-advertising-ban-with-deposit-limits-and-affordability-checks) |
| **Belgium** | Most gambling advertising banned (2023 decree, phased in); minimum gambling age 21; bonuses banned | [Andersen](https://www.be.andersen.com/en/news/gaming-advertising-regulation-in-belgium); [iGB](https://igamingbusiness.com/legal-compliance/belgium-implements-new-rules-banning-all-under-21s-from-gambling/) |
| **France** | Advertising is allowed but tightly supervised by the ANJ. We could not verify the details of the 2023 influencer law. ⚖️ | [iGaming Expert](https://igamingexpert.com/news/affiliates/french-regulator-balance-advertising-debate/) |

**Takeaway:** a betting-tips or affiliate product would need country-by-country work. A **fantasy and analytics subscription with no bookmaker links** can launch UK-first and extend to the EU with far less legal friction (VAT on digital services and GDPR still apply ⚖️).

### 3.7 Questions for a lawyer ⚖️
1. Confirm that an advice-only paid subscription needs no Gambling Commission licence.
2. Does showing bookmaker odds next to our probabilities, with no affiliate link, count as gambling advertising under CAP 16?
3. What proofing standard satisfies CAP for a model's track record (a public hash or git commit, or a third party)?
4. What does the DMCC subscription regime require for our checkout (monthly and season pass)?
5. Can we use driver names, photos and team logos at all? (FOM and team trademarks are a separate IP question.)
6. Do any EU markets restrict *selling betting advice* itself, not only advertising?

---

## 4. Pricing and willingness to pay

| Product type | Typical price | Free/paid split | Source |
|---|---|---|---|
| F1 fantasy tools | **€2.50–€6/mo** (yearly billing); Pro also monthly on Patreon | Big free tier (calculator, budget builder, live scoring, simulations). Paid tiers add personal analysis and league tools. | [F1 Fantasy Tools](https://f1fantasytools.com/pricing) |
| F1 AI prediction sites | Free; "Pro" upsell (price unknown) | Core predictions free | [f1predictions.ai](https://f1predictions.ai/predictions); [F1 Predictor AI](https://f1predictor.me/) |
| Odds comparison plus AI tools | **£0 now** (was £29.99–£34.99/mo) | Everything free, funded by affiliate deals | [Oddschecker](https://www.oddschecker.com/tips/horse-racing/20261028-oddscheckerplus-is-now-free-unlimited-odds-comparison-expert-betting-tips-return); [SBC News](https://sbcnews.co.uk/affiliatenews/2026/10/23/oddschecker-makes-oc-tools-free-closing-subscription-service) |
| US model-driven betting apps | ~$24.99/mo, ~$120–$200/yr | Free tier plus Pro | [App Price Lab](https://apppricinglab.com/app/apple/1083677479); [AIGearBase](https://aigearbase.com/tool/dimers) |
| UK generic tipsters | £10–£40/mo; £170–£300/yr | Trial month at a discount, then full price | [QuickFire](https://bmsmarketing.thrivecart.com/bt---quickfire-winners); [TipMan](https://www.tipmantips.com/premium-sports-betting-tips/); [Beat the Bookmakers](https://podosfairo.gumroad.com/l/beat-the-bookmakers) |
| Value-bet software | ~€89–€199/mo; some free plans | Trial, then subscription | [comparison site (unverified)](https://hiltonsoftware.pages.dev/compare/rebelbetting-vs-trademate-sports); [blog (unverified)](https://www.theleaguepaper.com/guest-blogs/399062/football-value-bets-how-to-beat-the-bookies-with-smart-betting/) |
| Tipster vetting | ~$139.99/yr | 90-day money-back guarantee | [BettorEdge (affiliate)](https://bettoredge.com/post/tipster-platforms-comparison) |

**Proposed pricing for Pitstop Oracle (a hypothesis to test, not a decision):**
- **Free:** the weekend forecast (top-5 win probabilities), the public track-record page, and the basic fantasy expected points. The track record must be free, because it is our marketing.
- **Paid "Season Pass"** at about **£19–£29 per season** (about £2–£3 a month over a ~9-month season), or about £3.99 a month. This fits between F1 Fantasy Tools (€30–€72 a year) and the generic tipsters. It includes the full probability tables (podium, points, DNF once it is good enough), the optimizer with our race probabilities, what-ifs, and the market-check view.
- Charge bettor-level prices (£10–£25 a month) **only if** G1 and the shadow record show that we match or beat the market.
- F1 is seasonal: offer a season pass rather than a rolling monthly plan, so the off-season doesn't drive cancellations. This also simplifies the DMCC renewal reminders (one renewal a year).

---

## 5. Distribution: where these users gather

| Channel | Who is there | Evidence | Notes |
|---|---|---|---|
| **r/formula1** | Mass F1 fans | About 389k new members in the past year ([Hive Index](https://thehiveindex.com/communities/r-formula1/)); its Discord has ~114k members ([Discord](https://discord.com/servers/r-formula1-177387572505346048)) | Strict self-promotion rules (check them). Best approach: share genuinely interesting analysis posts, not ads. |
| **r/fantasyF1 / r/F1Fantasy** | Fantasy players | Active weekly "Rate my Team" threads ([redlib mirror](https://redlib.ssps.io/r/fantasyF1)) | Strong fit for an expected-points table each weekend |
| **F1 Fantasy mini-leagues** | Fantasy players | Media outlets run public leagues ([GP247](https://www.grandprix247.com/formula-1-news/join-the-gp247-f1-fantasy-league-this-season); [FanAmp mini-leagues](https://www.fanamp.com/articles/mini-leagues-thrive-in-another-successful-season-for-f1-fantasy-2025)) | Run a "Pitstop Oracle vs the community" league in which our optimizer's team plays publicly. This is a live, honest demo. |
| **F1 Fantasy Tools Discord / Patreon** | Engaged fantasy players | Paid tiers include a Discord role ([pricing](https://f1fantasytools.com/pricing)) | A possible partnership: our race probabilities as an input to their tools |
| **Fantasy forums** (FISO, Autosport forums) | Long-time UK fantasy players | [FISO](https://www.fiso.co.uk/fanteams-formula-fantasy-2025-season-game/); [Autosport forum league](https://forums.autosport.com/topic/227773-f1-fantasy-league-2025-season) | Older, UK-heavy, analytical audience |
| **Podcasts** | Bettors and fantasy players | [Sporting Life Best Bets](https://www.sportinglife.com/racing/news/podcast-weekend-best-bets/170838), [F1 Money Gang](https://www.podengine.ai/podcasts/f1-money-gang-podcast), [Apex Insiders](https://apexinsiders.podbean.com/page/4/) (F1 betting, fantasy and DFS) | Guest spots offering "the model's view this weekend" |
| **X (Twitter)** | F1 data and stats community, tipsters | Social tipsters are common, and so are fake results ([Sporting Life guide](https://sportinglife.com/free-bets/guides/advanced/how-to-identify-a-good-tipster)) | A weekly forecast card plus a public scorecard. The honesty angle stands out here. |
| **Smart Betting Club** | Serious UK bettors | Independent tipster reviews ([Gambling.com](https://www.gambling.com/uk/news/should-you-pay-for-tips-the-knowledge-ep1-pete-ling-of-sbc-2091000)) | Later: ask them to track us, only if G1 passes |

**Launch-channel shortlist (in order):**
1. **r/fantasyF1, plus a public F1 Fantasy mini-league** run by the optimizer. Cheap, honest, and repeats every race weekend.
2. **A weekly "model vs grid vs result" post** on X and r/formula1 (an analysis format, not an ad), linking to the free track-record page.
3. **Fantasy podcast and Discord partnerships** (F1 Fantasy Tools community, FanAmp-style creators).
4. **Bettor channels** (betting podcasts, SBC) **only after G1** and with legal sign-off on the wording.

---

## 6. Positioning options

Plain-English model terms used below. *Log-loss* scores how good the probabilities are: lower is better, like a lap time. *De-vigged market* means the bookmaker or Betfair odds turned back into probabilities, with the bookmaker's built-in margin (the *vig*) removed. *Calibration* means that when we say "30%", it happens about 30% of the time.

### Option A: "Value-odds model"
*"Our probabilities vs the bookies': see where the price looks wrong."*
- **Pros:** the highest willingness to pay (£10–£40 a month, section 4). A clear, measurable promise. A natural fit for the UK bettor market chosen on 2026-11-09.
- **Cons:** if our model is worse than the market, the product is actively harmful to users, who would lose money following it. The most regulatory exposure (CAP tipster rules; affiliate pressure). Competes with free Oddschecker "value bets" and F1's own partner content. F1 has few markets and ~24 races a year, so proving an edge takes a long time.
- **Model accuracy needed:** post-qualifying winner log-loss **at or below the de-vigged market** (Betfair or sharp-bookmaker closing odds) over a meaningful sample, with good calibration. Also a positive return at available prices in a proofed live test. **Today: unknown, and unlikely**, given we only just beat the grid (1.422 vs 1.466, and slightly worse in 2026: [`ROADMAP_2027.md`](ROADMAP_2027.md)).

### Option B: "Transparent F1 analytics plus fantasy helper" ✅ recommended (subject to G1)
*"The F1 model that shows its homework, and helps you pick your fantasy team."*
- **Pros:** the biggest audience (2.75M+ fantasy teams). Lowest regulatory risk; can be open to all ages outside betting features. Uses what exists today (Fantasy Lab, optimizer, Race Explorer, Model Performance). An honest track record differentiates us from the free AI predictors. Expands easily to the EU. Leaves the door open to Option A.
- **Cons:** a low price ceiling (fantasy tools cost €2.50–€6 a month). Free competitors already exist. Needs good fantasy UX, not just a model. Less fit with the "UK bettors" headline. Bettors become a secondary audience, served by an informational market-check view.
- **Model accuracy needed:** beat the starting grid on walk-forward data, with **calibrated** probabilities (gate G3), plus expected fantasy points that beat a simple baseline (for example, "points from last 3 races"). We **don't** need to beat the betting market. **Today:** close. Pre-qualifying beats prior form clearly; post-qualifying only just beats the grid.

### Option C: "Track-record-first tipster"
*"Proofed F1 picks with a public record."*
- **Pros:** the clearest story for bettors. The proofed record itself is the product. Smart Betting Club-style validation is possible.
- **Cons:** needs **years** of results. With ~24 races a year and a handful of bets per race, a profitable record is mostly luck for a long time. Can't be ready by March 2027. Tipster marketing has the worst reputation and is the most heavily policed (ASA rulings). One bad run destroys the brand.
- **Model accuracy needed:** a **positive return on investment at available odds** over hundreds of proofed bets, which implies beating the market on the races we pick. That is a stricter standard than Option A.

### Comparison

| | A: value-odds | B: analytics plus fantasy | C: tipster |
|---|---|---|---|
| Audience size | Small | **Large** | Small |
| Price per user | **High** | Low | Medium–high |
| Regulatory load | High | **Low** | High |
| Model bar | Beat the market | **Beat the grid, calibrated** | Beat the market plus profit |
| Ready by March 2027? | Only if G1 passes | **Yes, likely** | No |
| Uses existing features | Partly | **Mostly** | Little |

### Recommendation
**Launch as Option B**, with three design rules that keep Option A open:
1. **A public, timestamped track record from the first 2026 shadow race** (log-loss vs grid; later vs market). It is the trust engine in every option.
2. **A free market-check view**: our probability next to the de-vigged market probability, framed as information ("where the model and the market disagree"), with no bet buttons, no affiliate links and an 18+ gate. It serves curious bettors honestly and builds the evidence for Option A.
3. **Upgrade trigger:** if G1 (and then the live record) shows our post-qualifying log-loss at or below the de-vigged market, add a paid "value" tier for bettors (Option A) after legal review.

**This depends on G1.** If workstream B finds that we already match the market (unlikely), revisit A as the headline. If we lose clearly to the market, B is the only honest choice, and the market-check view should stress "the market is usually right; here is where we differ".

---

## 7. Open questions for Siddharth
1. **The headline audience.** The 2026-11-09 decision put UK bettors first, but the evidence favours fantasy players for launch. Are you comfortable with fantasy-first marketing, with bettors as a secondary audience (served by the market-check view)?
2. **Affiliate revenue.** Are you willing to forgo bookmaker affiliate income at launch, to stay independent and simplify compliance?
3. **Legal budget.** Can we fund a short gambling and advertising law review (the questions in section 3.7) before January 2027?
4. **Brand assets.** Are you comfortable without official F1, driver or team imagery (trademark and IP risk)?
5. **Price test.** Should we test a £19–£29 season pass against about £3.99 a month with a waitlist and a "fake door" page before building payments (payments come after launch per the roadmap, so maybe just a waitlist)?
6. **Partnerships.** Should we approach F1 Fantasy Tools or a fantasy podcast to share our probabilities (distribution in exchange for data)?
7. **Market data.** Which odds source will the market-check view use (Betfair Exchange API, an odds API)? This depends on workstream B's choice, and its licence terms may limit public display.
8. **Validation.** Can you name 5–10 F1 fantasy players or bettors we could interview before the MVP scope freezes (G2, 15 December)? Desk research can't replace talking to users.

---

## 8. Glossary
| Term | Meaning |
|---|---|
| **Affiliate** | A website paid by a bookmaker for sending it customers, either as a flat fee per sign-up (*CPA*, cost per acquisition) or as a share of the customer's losses (*revenue share*) |
| **ASA / CAP / BCAP** | The UK Advertising Standards Authority enforces the CAP Code (non-broadcast ads: web, social, print) and the BCAP Code (TV and radio) |
| **Betfair Exchange** | A betting marketplace where people bet against each other rather than against a bookmaker. Betfair takes a commission on winnings. |
| **Calibration** | When the model says 30%, the event happens about 30% of the time. Like a tyre-wear prediction that matches the real degradation. |
| **Chips (F1 Fantasy)** | One-off power-ups in F1 Fantasy (for example unlimited transfers for one race) |
| **Churn** | Subscribers cancelling. In F1 it spikes in the off-season. |
| **DMCC Act** | The Digital Markets, Competition and Consumers Act 2024, a UK law that includes the new subscription contract rules |
| **De-vig / de-vigged odds** | Converting odds into probabilities and removing the bookmaker's margin so they add up to 100% |
| **Exchange trader** | Someone who backs and lays on an exchange to profit from price movements, often during a race |
| **Freemium** | A free basic product with paid upgrades |
| **Gambling Commission (UKGC)** | The UK regulator that licenses gambling operators |
| **GSGB** | The Gambling Survey for Great Britain, the Commission's official participation survey |
| **LCCP** | The Gambling Commission's Licence Conditions and Codes of Practice: rules for licensed operators |
| **Log-loss** | A score for probability forecasts: lower is better. It punishes confident wrong calls heavily, like a big penalty for crashing out from pole. |
| **Overround / vig / margin** | The bookmaker's built-in profit. The implied probabilities add up to more than 100% (for example 110%). |
| **Proofing** | Lodging a tip with an independent third party *before* the event, so the track record can't be edited afterwards |
| **Responsible gambling (RG)** | Measures that reduce gambling harm: age checks, warnings, links to help services |
| **Sharp bettor** | A skilled, often model-driven bettor whose bets move prices. "Sharp" markets react to them quickly. |
| **Starting price (SP)** | The official odds at the start of the event; the standard benchmark for tipster profit claims |
| **Strong appeal (to under-18s)** | The CAP test for banning gambling ads that are likely to attract children |
| **Tipster** | A person or service selling betting selections |
| **Value bet** | A bet whose odds imply a lower probability than you believe is true. It is a long-run edge, never a guaranteed win. |
| **Walk-forward evaluation** | Testing a model only on races after its training data, the way it would be used in real life |

---

## Method and limits
- Web desk research on 2026-11-09. No interviews, surveys or paid reports.
- Market-size numbers for F1 betting are poor. Public F1-specific betting turnover and Betfair F1 volume figures were not found from trustworthy sources.
- Several prices come from third-party trackers or affiliate pages (marked "unverified"). Check them on the vendors' own sites before quoting them externally.
- Regulation summaries rely partly on law-firm commentary and trade press. ⚖️ items need professional advice.
