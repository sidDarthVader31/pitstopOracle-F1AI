# Ops plan: hosting, scheduled runs, public ledger, monitoring (proposal)

**Workstream E, 2026-11-10.** Proposal only: nothing here has been signed up for, created,
pushed or deployed. Every account, cost and visibility change is the owner's decision
(listed at the end). Prices and free-tier limits change; check them before deciding.

**Labels:** [today] exists in the repo · [proposed] this document · [later] after launch.

## What has to run

| Job | When | Runtime | Writes |
|---|---|---|---|
| App (Streamlit, 6 pages) [today] | Always on | ~1 CPU, ~1 GB RAM is plenty (sklearn models + 6,000-sample simulations, cached) | Nothing (read-only) |
| `race_weekend.py before` [today] | Once per weekend, ≥ 1 h before the team lock (we aim for ~10 h) | ~1–3 min incl. ingest | `ml/ledger/`, `ml/models/`, `ml/data/` |
| `race_weekend.py update` [today] | After qualifying, and when the official grid is out | < 1 min | `ml/ledger/` |
| `race_weekend.py after` [today] | A few hours after the race | < 1 min | `ml/ledger/scores.jsonl`, calibration history |
| Prices entry [today, manual] | Thursday | 15 min of the owner's time | `data/manual/fantasy_prices.csv` |

The single hard deadline is the **pre-lock snapshot**: a missed lock is the worst failure
(MVP_SPEC §10.1 target: 6/6 on time). Everything else can be late without hurting trust.

## 1. Hosting the Streamlit app

| Option | Cost / month | Pros | Cons |
|---|---|---|---|
| **A. Streamlit Community Cloud** | Free | Zero ops; deploys from a GitHub branch on every push (so a new ledger commit appears automatically); HTTPS included | `*.streamlit.app` address only (no custom domain); limited RAM; apps sleep when idle and take a while to wake; no persistent disk; can't inject an analytics script into the page; needs the repo connected (public repo, or its private-app allowance) |
| **B. Small VM** (e.g. Hetzner CX22 ≈ €4–5, DigitalOcean/Lightsail ≈ $5–6) with Docker + Caddy | ≈ €5 + domain (~£10/yr) | Custom domain; always on; cron for the weekend runs on the same box as a fallback; persistent disk for privacy-friendly server-side analytics; full control of the page `<head>` | You patch and back up the box; ~2 h setup; a single point of failure unless the weekend runs live elsewhere |
| C. Container PaaS (Render, Railway, Fly.io) | ≈ $5–25 | Custom domain; managed TLS and deploys; some have cron jobs | Cost creeps; persistent disks are add-ons; cold starts on the cheapest tiers |
| D. Hugging Face Spaces | Free | Streamlit support | Odd fit for a product site; same limits as A |

## 2. Scheduling the weekend runs

**Proposed: GitHub Actions on a schedule, with the ledger committed back to the repo.**

- One workflow (`.github/workflows/race-weekend.yml`, [proposed]) runs **every hour from
  Thursday to Monday on race weeks**. A small `race_weekend.py auto` command ([proposed],
  not built) reads `weekend_schedule.csv` and decides what is due: `before` from lock −12 h,
  `update` once qualifying/grid data appears, `after` once results are in. Running hourly
  and letting the script decide makes the schedule self-correcting (GitHub's cron can be
  delayed at busy times, and a failed run is simply retried an hour later).
- The job commits `ml/ledger/` (and models/data) with a bot identity and pushes. Branch
  protection forbids force-pushes and deletions on the ledger branch, so the git history
  itself becomes the second, independent timestamp.
- Cost: free for public repos; private repos have a monthly free-minutes allowance that
  this (~5 short runs a weekend) fits easily.
- **Fallback:** the VM's cron (option B) or the owner's laptop
  ([`RUNBOOK_RACE_WEEKEND.md`](RUNBOOK_RACE_WEEKEND.md)) runs the same command. Two
  writers can't corrupt the ledger: writes are exclusive-create and every run verifies the
  ledger first; the worst case is two snapshots a few minutes apart, both honest.

## 3. Where the ledger lives (public and append-only)

[today] `ml/ledger/` in this repo: content-hashed files, hash-chained `index.jsonl`,
`python race_weekend.py verify`. To make it **public and tamper-evident**:

1. **A public git repository** holding the ledger. Either make this repo public, or (better,
   given MVP_SPEC §7.3: the repo name contains "F1") publish only `ml/ledger/` to a separate
   public repo such as `pitstop-oracle-ledger`, pushed by the weekend job. Protect its main
   branch: no force-push, no deletion.
2. **Show the integrity check in the app** [today]: Track Record re-verifies every hash on load.
3. **Optional, cheap extra proof** [proposed]: publish each pre-lock snapshot's SHA-256
   somewhere we don't control, e.g. in the weekly pre-lock email/social post, or an
   OpenTimestamps proof. Anyone can then check the file existed before the lock.
4. [later] Object storage with write-once retention (e.g. Backblaze B2 or S3 Object Lock) if
   the ledger grows beyond what git handles comfortably (it won't for years: ~30 KB per
   snapshot, ~5 snapshots a weekend).

## 4. Monitoring and alerts

| Check | How | Alert |
|---|---|---|
| Pre-lock snapshot on time | The hourly job checks at lock −3 h: is there a `pre_quali` snapshot with `published_before_lock: true`? If not, runs `before` immediately and alerts | Email from the failing GitHub Action, plus a push to the owner's phone (e.g. ntfy.sh or a Telegram bot: free) |
| Job failure | GitHub Actions emails on failure | Email |
| Ledger integrity | `race_weekend.py verify` as the first step of every job and in CI on every push | Job fails |
| App up | Free uptime monitor (UptimeRobot / Better Stack free tier) on the home page | Email/push |
| Stale data | The app already warns visitors when qualifying is in but no post-qualifying snapshot exists | On-page |
| Tests | `pytest ml/tests` in CI on every push [proposed] | PR check |

## 5. Privacy-friendly analytics

Streamlit has no analytics, and on Community Cloud a script in `st.components.html` runs in
an iframe and can't see page views. The metrics we need (MVP_SPEC §10: lock-day visitors,
returning visitors, Check-my-team usage, "did this help?" poll) are few, so:

- **Recommended:** **server-side event counts** inside the app: on each session, count the
  page and whether Check my team was used, as aggregate daily counters (no IP addresses, no
  cookies, no user IDs). Needs a writable store, so it fits option B (SQLite on the VM) or a
  tiny free database. No consent banner needed for aggregate counts without identifiers
  (to be confirmed in the legal review ⚖️).
- Alternatives: Plausible (≈ $9/month, cookieless, EU-hosted), Umami (self-hosted free on
  the VM, or its cloud free tier), GoatCounter (free for small sites). All need control of
  the page `<head>` (option B) to count properly.
- Email capture: the app has a placeholder form that stores nothing. Pick a tool with double
  opt-in and one-click unsubscribe (e.g. Buttondown, MailerLite, or a Tally form). UK GDPR:
  consent checkbox (already in the form), privacy note (already on How it works).

## 6. Cost summary

| Setup | Monthly | Yearly |
|---|---|---|
| A + GitHub Actions + free uptime monitor | £0 | £0 |
| B (VM) + domain + GitHub Actions + self-hosted analytics | ≈ £5 | ≈ £70 |
| B + Plausible + paid email tool at small volume | ≈ £15–25 | ≈ £180–300 |

## 7. Recommendation

1. **Now → end of 2026 (shadow challenge):** run the weekends **by hand** from the runbook
   (automation isn't built yet). Commit and push the ledger after each run if you want the
   record public now.
2. **December–January:** build the `auto` command and the GitHub Actions workflow;
   create a **separate public ledger repo**; add CI (`pytest`, `verify`). Deploy the app on
   **Streamlit Community Cloud** for interviews and usability tests: free and zero ops.
3. **By launch (late February 2027):** move the app to a **small VM with a custom domain**
   (no "f1" in it), Caddy, server-side counters, and VM cron as the fallback runner. Keep
   GitHub Actions as the primary runner and publisher of the ledger. About £5/month.

Why: the ledger and scheduled runs need no server at all (GitHub does both, for free, with
an independent timestamp). The app is the only always-on piece; Community Cloud is enough
to learn from users, and a VM buys the custom domain and analytics we need for launch.

## Owner decisions needed

1. Make the ledger public now (push), and where: this repo or a separate public ledger repo?
2. Streamlit Community Cloud account (connects to GitHub) for the Dec–Feb test app?
3. Budget for a VM and domain (≈ £70/year), and the domain name.
4. Analytics: server-side counters (recommended) or a hosted tool?
5. Email tool for the waitlist.
6. Who gets alerts (phone push / email) on race weekends?
7. Data-source terms for a public app (Jolpica, OpenF1, Open-Meteo's non-commercial tier) ⚖️.
