# Stock Predictor — Project Design

A Python service that predicts the best intraday opportunities for the current
session — which symbols, for how long, and for what gain — and serves that
over a low-latency REST API, a live dashboard and an on-demand summary.

It is a sibling of `stock-analyzer`, which it reads from and never writes to.
stock-analyzer records what *was* available; stock-predictor tries to say what
*will be*.

This document says what is predicted and why it is built the way it is.
[ARCHITECTURE.md](ARCHITECTURE.md) shows the modules, the data flows, where
data is stored and what each session writes. The [README](../README.md) is the
user's guide.

---

## 0. Read this part first

Four facts about the available data shape everything below. Pretending
otherwise would produce a design that looks impressive and does not work.

**Minute history cannot be bought back.** The provider caps intraday history
at roughly three months, so the data grows one session a day from when
stock-analyzer started recording. It now holds **111 sessions** of minute
candles (2026-06-02 to 2026-11-09) for **62 symbols**: 2.3 million candles.
Daily candles in its views cover the same sessions. The design trains on what
exists and is built to accumulate forward.

**There is far more training data than there are days — provided the model is
trained at the right granularity.** Symbols on a day share one market, but not
much of it: over the first 66 sessions, the market factor explained only
**16.9%** of daily return variance across the universe, and mean
cross-sectional dispersion was **0.96%** against a mean market move of
**0.47%**. Symbol-specific movement is roughly twice the market-wide movement,
so symbols are much closer to independent than to copies of one another.

The data available depends entirely on what a training example *is*:

| Training unit | Examples available |
|---|---|
| Minute candle, forward-looking label | **2,303,541** |
| Non-overlapping 30-minute episodes | **~74,900** |
| Symbol-session | **6,304** |
| Session (market regime only) | **111** |

Only the last row is genuinely scarce, and it constrains only features that are
identical for every symbol on a day: breadth, market return, dispersion. Those
should be used sparingly. Everything symbol-specific has orders of magnitude
more. So the model is trained per candle, where the data is, and its output is
*aggregated* into the answer (section 5). The granularity of the answer does
not have to be the granularity of the training.

**Fundamentals exist only as a snapshot.** stock-analyzer fetches screener.in
ratios, quarterly results and shareholding, and runs a daily CANSLIM screen.
The ratios are one current snapshot per company, and the quarterly results
carry no publication dates, so none of it can be used on past sessions
without look-ahead. It is shown and measured, not modelled (section 3.4).

**So far, nothing beats a simple volatility screen.** Section 6.1 has the
numbers. That is the outcome this design was built to detect early, and it
shapes how every answer is presented: each prediction is shown next to the
screen it has to beat, with the probability calibrated against what actually
happened.

---

## 1. What is predicted

Two kinds of answer, from the same features and the same per-candle model
family.

### 1.1 The daily ranking: the five best windows still open

stock-analyzer computes, for every symbol and session, the top non-overlapping
windows in which a gain was available: entry time, exit time, gain percent,
duration. That table, versioned by detector, is the ground truth.

The target is the **five best windows across the whole universe for the current
session, still open after the moment T**, ranked by gain. A prediction is five
rows of:

```
symbol · expected entry window · expected duration · expected gain %, with a range
```

That is deliberately three separate predictions with very different difficulty,
and treating them as one regression would hide that:

| Component | Method | Why |
|---|---|---|
| **Which symbols** | Ranking over the day's universe | Relative ordering is far more learnable than absolute outcome |
| **Gain %** | Quantile regression (p25 / p50 / p75) | A point estimate of a noisy quantity invites false confidence; a band is honest |
| **Duration** | Bucket (<15m, 15–60m, 60–180m, >180m) | Exact minutes are not predictable; the bucket is what changes a decision |

### 1.2 Fixed targets: will this stock make +X%?

A plainer question, asked of every symbol at any minute: **will the price
trade X% above the price now within the next H minutes**, or **before
today's close**? Optionally, it must get there before falling by a stop. The
answer for each symbol is:

- **a chance**: the model's probability, mapped to how often that probability
  came true on sessions the mapping was fitted without;
- **a band**: p25 / p50 / p75 of either the return at the end of the window,
  or the best gain available in it (the highest later price against the price
  now).

This is what `predict`, `summary`, `scripts/today.sh` and the dashboard show:
for example, "which stocks are expected to gain 0.5% or more before today's
close, and by how much". Its ground truth is the candles themselves, so it
needs no detector, and every finished session can be used.

### 1.3 The next session

The same question about a session that has not opened: will the stock trade
X% above **that session's open** before its close? The open is a price anyone
can buy at, through the pre-open auction; an overnight gap is not counted as a
gain. The model has one row per (symbol, session): the stock's full state at
15:00 the session before (the same 84 features), against the next session's
high, open and close. `scripts/tomorrow.sh` shows it.

### 1.4 Every answer is recorded and scored

Each run of `today.sh` and `tomorrow.sh` stores every stock's prediction:
chance, band, reference price, and the minute it used. After the session
closes, the prediction is scored against what happened, giving a scorecard
per session and a track record per stock. Before that, the held-out month
supplies one: how often, for each stock, a model that had not seen the month
said "50%+", and how often that came true.

---

## 2. Timing, and the leakage rule

This is where projects like this usually go wrong, so it is settled first.

A prediction is always made **as of a moment T**:

- features may use only data available at or before T;
- a ranking label counts only windows whose **entry is strictly after T**;
- a fixed-target label starts from the close of T's own candle and looks only
  at the candles after it.

Predicting at 09:30 that a window which opened at 09:20 was good is not a
prediction. It is reading the answer. `labels/targets.py` holds the ranking
rule in one place, and `leakage_audit()` re-checks it over real sessions.

**Every row depends only on candles at or before its own minute.** A
session's training frame is built once, to 15:00, and cut back to any earlier
prediction point. So any value taken at build time and copied to every row
would reach back into the earlier rows: a 09:30 row would know where the stock
was at 15:00. Every feature is computed per row, and a test builds a session
at 10:30 and at 15:00 and requires the shared rows to agree on every column.

**Daily values are as of the session before.** Moving-average distances, RSI,
trend score, 10-day range position and beta all come from sessions strictly
before the one being predicted. A view that holds only each symbol's latest
value (`v_range_context`, `v_symbol_beta`) would leak later sessions into
every past one, so those values are recomputed from daily candles. The 52-week
range position is dropped: stock-analyzer keeps only a current snapshot of it.

Three prediction points for the ranking, each a distinct model context:

| Point | When | What it can see | Honest expectation |
|---|---|---|---|
| **T0** | 09:00, pre-open | Prior sessions, overnight gap proxies | Weakest. Mostly a volatility ranking |
| **T1** | 09:30, after the opening range | T0 plus the open, first 15 minutes, gap, opening-range break | The most useful single call of the day |
| **Tn** | every 15 min to 15:00 | Everything so far that session | Narrowing, as fewer minutes remain |

The remaining-opportunity framing has a pleasant property: the label naturally
shrinks through the day, so the model learns that a 2% window is less likely
at 14:30 than at 09:30 without being told.

**"Now" is the latest minute most symbols have finished.** stock-analyzer
writes the session in progress a symbol at a time, every 150 s. The newest
minute is often held by only a few symbols, and ranking those few against each
other would distort every cross-sectional feature. Live predictions use the
latest minute at least 90% of symbols have finished, which runs two to three
minutes behind the clock.

**Windows end inside the session.** A fixed-horizon window that would run past
the last candle (15:29) has its outcome cut short, so such rows are not
trained on, and live answers stop at the last minute whose window fits (14:49
for 40 minutes). A window that runs to the close always ends there. Every
model is trained on rows up to 15:00 and is not asked about later minutes.

---

## 3. Data

### 3.1 From stock-analyzer

Read-only, over its published views rather than its base tables. The views are
a deliberate contract; the tables behind them are not:

- `v_intraday_candles`: finished sessions' minute candles;
- `v_intraday_merged`: the same plus the session in progress;
- `v_daily_candles`;
- `v_symbol_trend`, `v_support_resistance`, `v_market_breadth`,
  `v_range_context`;
- `v_opportunity_markers`, the ground truth for the ranking.

stock-analyzer writes a day's candles to `v_intraday_candles` only after the
close. During the session, its daemon records each finished minute, and
`v_intraday_merged` adds those to the finished sessions. Live reads take today
from there, leaving out the candle still forming. Every other input to a
feature comes from sessions before today.

Coupling is one-way and read-only. stock-predictor never writes to that
database; it keeps its own. The connection is opened read-only but not as
immutable, because the daemon writes all session long.

### 3.2 Ground truth and completed sessions

stock-analyzer versions its gain-window detector, and the series moves on:

| Detector | Symbols | Final sessions | Status |
|---|---|---|---|
| `topk-nonoverlap/highlow/v1` | 49 | 68, ends 2026-09-07 | no longer produced |
| `topk-nonoverlap/highlow/min0.5/v1` | 62 | 2026-06-02 → 2026-11-05 (partial on the last day) | **this project's default**; no longer produced |
| `topk-nonoverlap/highlow/min0.5/hold5/v1` | 62 | 2026-11-04 onwards | what stock-analyzer now writes |

`min0.5` keeps only windows of at least 0.5%, roughly where costs stop erasing
a move (section 6). `hold5` adds that the exit is at least five candles after
the entry. The ranking trains and evaluates only on **completed sessions**:
those with `final` labels under the configured detector
(`STOCK_PREDICTOR_DETECTOR`). That excludes the session in progress, which has
only `live` labels. Under the default it also ends the ranking's data at
2026-11-05. Switching to `hold5` changes what a window is, so it is an open
decision (section 11), not a fix.

Fixed targets need no detector, and use every finished session.

**The universe grew from 49 to 62 symbols**, and the newer symbols' history was
backfilled to 2026-08-07. Some were added because they topped a CANSLIM chart.
That is selection on recent performance, applied to the past, so it flatters
any strategy that favours them in the backfilled period (open question 2).

### 3.3 The day's own shape

Open, high, low so far, opening-range break, gap versus prior close, VWAP and
distance from it, volume against the same minute on other days, position within
the day's range. stock-analyzer computes these 26 features too, and pins them
with a shared fixture (`feature-parity.json`). This project reproduces them to
1e-9, so the two implementations cannot drift apart.

### 3.4 Screener data

stock-analyzer also keeps:
- screener.in fundamentals: P/E, P/B, ROE, ROCE, debt, growth, market cap,
  52-week position;
- quarterly results and shareholding;
- a daily CANSLIM screen: letters passed, RS rating, signal, distance from
  pivot;
- a daily bullish score;
- a next-day watchlist.

None of it is a model feature:

- **The fundamentals are one snapshot per company.** Used on past sessions,
  they would be look-ahead, so they are shown as context and labelled as
  untested.
- **CANSLIM runs, the bullish score and the watchlist carry dates.** They are
  joined point-in-time, using only what was published before each session's
  open, and measured against a target (section 4.2). The CANSLIM history began
  on 2026-10-26.

A point-in-time fundamentals source would go behind the existing interface,
which has a null implementation so everything else runs without it:

```python
class FundamentalScore(Protocol):
    def score(self, symbol: str, as_of: date) -> float | None: ...
```

Quarterly fundamentals move four times a year and intraday windows resolve in
hours. Expect them to shift the ranking *between* symbols slightly and to say
almost nothing about *today*.

### 3.5 News — stretch

Deliberately last. Headline sentiment is the feature most likely to leak
(publication timestamps are unreliable), most likely to be stale, and hardest
to evaluate. Not started until everything above is measurably working.

---

## 4. Features

Three groups, and the middle one is what makes a ranker work. 84 columns in
all with the candidate inputs off.

**Per-symbol**: the 26 contract features (returns over several windows,
volatility, ATR, VWAP distance, volume z-score against the same time of day,
opening-range distances, gap, event flags), plus daily RSI, distances from the
5/20/50-day averages, trend score, 10-day range position and beta, all
point-in-time (section 2).

**Cross-sectional**: every per-symbol feature also enters as its **rank and
z-score within that minute's universe**. The question is "which of these is
best right now", so relative standing matters more than absolute values, and
normalising strips out the market-wide move that would otherwise dominate.

**Regime**: the previous session's breadth, mean return and dispersion, day of
week, and each row's own minutes since the open. These are identical for every
symbol at a minute and exist to let the model condition on the kind of day it
is. They have only 111 observations, so they are kept few.

### 4.1 Candidate screening inputs — kept as parameters, measured before use

Seven inputs were proposed for picking stocks and tested elsewhere first. The
opening check was tested on 5,982 stock-sessions across these 62 symbols. The
other inputs were tested on about 182,000 stock-sessions of the whole liquid
market, against **next-day** moves. Most showed no reliable effect.

They are kept as named parameters in `params.py`, all defaults in
`params.example.toml`, so they can be measured on **this** project's target
rather than accepted or dropped on faith. Each input has two independent
switches:

- **`feature`** adds the input's value as a stage-1 column. This is soft: the
  trees may ignore it. It takes effect on retraining, and the model card
  records the switches so serving can match them.
- **`filter`** drops symbols that fail the input's rule before ranking. This
  is hard: it changes which symbols can be picked at all.

Both are **off by default**. A rule over a value that is not available yet
(too little daily history, or an opening check whose time has not come) does
not apply, and the symbol is kept. `stock-predictor ablate` measures them. The
figures below use the ranking target (the net-of-cost best window still open
after T) at 09:45, so the opening check is known, over 87 development
sessions. Intervals are 95% bootstraps over sessions.

| Input | Parameters (default) | Earlier finding (next day) | Measured here | Status |
|---|---|---|---|---|
| **Opening check** | `check_time` 09:45, `opening_range_minutes` 15, `allowed_range_states` [above], `require_above_vwap` true | Range and VWAP state at 09:45 told nothing about the rest of the day | **Informative, in reverse.** IC −0.15 [−0.19, −0.11]. The 14% above the range and above VWAP have 0.18 pp *less* remaining gain. Below-range, below-VWAP symbols have the most. As a filter it cuts both baselines' top-5 by ~0.4 pp | Off. Never use it as a breakout filter. As a feature it is the one worth a retrain test |
| **Run-up cap** | `lookback_days` 5, `max_pct` 10 | No reliable effect | The cap rarely binds (98% pass). The 2% over it have +1.2 pp larger windows *and* a 32% adverse rate against 8%. That is a volatility signal. As a filter it costs 0.10 pp | Off |
| **RSI limits** | `period` 14, `min` 30, `max` 70 | No reliable effect | IC +0.07 [+0.04, +0.10]: small. No pass/fail difference. As a filter it costs 0.16 pp on `yesterdays_leaders` | Off. `rsi14` is already a feature |
| **Volume** | `lookback_days` 20, `min_relative_volume` 1.0 | No reliable effect | IC +0.01 [−0.02, +0.05]. Nothing | Off |
| **Close location** | `min` 0.5, `max` 1.0 (prior-day close in its range) | No reliable effect | IC +0.04 [+0.01, +0.07]: barely there. No filter effect | Off |
| **Score order** | `rank_by` score, `adverse_move_pct` 2.0 | The strongest fifth by score is just more volatile: 6.6% fall more than 4% next day, against 4.4% for the market | **Confirmed in kind.** By ATR quintile, net gain rises 0.63 → 1.11% while the adverse rate rises 5% → 14% | `score` stays the default. `risk_adjusted` is available. `adverse_rate` is reported for every strategy |
| **Medium-term momentum** | `lookback_days` 120, `max_dist_from_high_pct` 10 | Points the right way over 10–20 sessions, but weakly. CANSLIM covers that horizon | **Not measurable yet.** Daily history is shorter than 120 sessions. A 60-day proxy: IC +0.03 [−0.03, +0.08], no top-5 effect | Off. Re-measure once 120 sessions exist for the symbols in question |

How to read this:

- **The target here is an upper bound.** It is the best window still
  available, not a trade anyone took. Anything that tracks how much a symbol
  moves will look informative against it. The adverse rate is there to catch
  that, which is how the run-up and score-order rows read.
- **Six inputs are tested at once**, each against roughly 62 symbols a day. One
  interval excluding zero by chance is plausible. RSI and close location are
  within that noise. The opening check, at IC −0.15 with a tight interval, is
  not.
- **The two studies do not contradict each other.** The earlier tests asked
  about next-day direction. This one asks how much upside is left today. The
  opening check can say nothing about the first and something about the second.

**Rule for switching one on.** A filter is switched on only if it raises top-5
net gain on the development sessions, with an interval excluding zero, and does
not raise the adverse rate. A feature is switched on only if a model retrained
with it beats the model without it on walk-forward folds. Either is then
confirmed once on the held-out month, and dropped if it does not hold there.
None of the seven meets the filter rule.

### 4.2 Screener signals — measured, not used

`screener-evidence` puts every symbol at every 15-minute point in or out of
each group, using only what was published before that session's open. It
compares the target's hit rate between the two. Among a model's top picks on
sessions it was not trained on, it asks whether the signal would change the
call. On "+1% within 40 minutes":

| Signal | Sessions | In the group vs the rest | Among the model's top 5 |
|---|---|---|---|
| Bullish score in the top fifth (previous session) | 89 | 5.2% vs 3.0% | 16% vs 19% |
| On the watchlist made after the previous session | 89 | 7.1% vs 3.6% | 18% vs 18% |
| CANSLIM RS rating 80+ (latest run before the open) | 8 | 7.1% vs 4.0% | 21% vs 9% |
| 4+ of 7 CANSLIM letters passed | 8 | 9.1% vs 4.0% | 26% vs 9% |

On its own, a high bullish score or a watchlist place has meant about twice
the chance. Among the model's picks it adds nothing: the model already sees
what those signals see. CANSLIM looks strong in both columns, but eight
sessions are too few to read. A difference is read only with at least 100
symbol-days in the group: a lift of 1.2 or more "supports", 0.83 or less is
"against".

---

## 5. Models

Two stages, and the split between them is the most important structural decision
in the project.

### Stage 1 — a per-candle model, trained where the data is

```
forward     gradient-boosted trees, one row per (symbol, minute)
            label: what happens over the window that starts after this minute
            ~2.3M rows, or ~0.7M at every 3rd minute (every 5th scored within
            noise of every minute)
```

The label is a barrier question over a window that starts after each minute:

- **ATR barriers** (the ranking's default): an upper barrier at +u, a lower
  at −d, a time barrier at H minutes. The barriers are scaled by recent ATR
  so they mean the same thing in calm and volatile conditions. This is the
  labelling stock-analyzer's model contract describes, so the two share a
  definition.
- **A fixed target**: +X% within H minutes, or before the close; with a stop,
  the target must come before −Y%.

Windows are clock minutes read from the session's full candles, so a minute
with no trades does not stretch one. When one candle reaches both barriers,
the order inside it is unknown, and the row counts as stopped.

Two heads share the feature matrix:

- a **classifier** for P(the upper barrier or target is reached first);
- a **quantile regressor** for a p25 / p50 / p75 band: by default, the return
  at the end of the window; with `--band max-gain`, the best gain available in
  it.

A fixed-target model also carries a **calibration**. The classifier is fitted
without the last 20 training sessions. Its probabilities on those sessions are
mapped, isotonically, to how often they came true. The model is then refitted
on everything and keeps that map. Every answer shows both numbers, and the
model card keeps the table.

The training window is the full history by default. `--recent-days N` trains
on only the last N days, for a model that follows the current regime more
closely. `--minute-step K` keeps every K-th minute. Neither the last month nor
every minute instead of every 3rd changed held-out results measurably
(section 6.1). The model's name and card record both, and the three quantiles
are sorted per row, since separately fitted ones can cross.

This stage sees millions of examples. It is where any real signal will be
found, and it is trained without ever knowing what a "top five" is. LightGBM is
preferred. scikit-learn's histogram GBDT, the same family, is used when
LightGBM's native runtime is missing, and the model version records which.

### Stage 2 — aggregate into the ranking

The daily ranking is a deterministic reduction of stage 1, not a second
learned model:

```
for each symbol, at prediction point T:
    expected best window  = max over remaining minutes of the forward model's
                            predicted move, discounted by its own probability
    rank the universe by that number, take five
```

Magnitude and duration come from the same stage-1 outputs: quantiles of the
predicted move for the gain band, and the horizon at which the upper barrier is
most often struck for the duration bucket.

How that number orders the universe is a parameter (`score_order.rank_by`,
section 4.1). `score` ranks on it directly. `risk_adjusted` divides it by the
symbol's ATR % first, so a symbol does not reach the top merely by being the
most volatile. Any active filters are applied before the top five are taken.

Only a thin calibration sits on top, mapping stage-1 scores onto realistic
gain expectations with a handful of parameters. That is the one place the
session count genuinely binds, and fitting a few parameters on ~100 days is
defensible where fitting a full ranker on them would not be.

### Baselines that must be beaten

Built and measured **before** any model, because without them a metric is just a
number:

1. **Yesterday's leaders**: the five that had the biggest windows last session.
2. **Widest 20-day ATR** (`highest_volatility`): volatility genuinely predicts
   the *size* of a move, so this is the honest competitor for the ranking.
3. **Most volatile right now**: the five with the widest 14-minute ATR as a
   share of price. A fixed target is mechanically easier for a volatile stock,
   so this is the one a fixed-target model has to beat.
4. **stock-analyzer's time-of-day prior** (`hot_window`).
5. **Random five**: the floor.

---

## 6. Evaluation

**Split by day, never within one.** All symbols of a session go to the same
fold. Purged walk-forward with an embargo of one session either side of each
boundary, since labels look forward within a day. Splitting by day is about
**leakage**: two minutes from one session share label horizons and cannot sit
on opposite sides of a split. It says nothing about how many training examples
exist. There are ~100 days of *boundaries* and millions of *examples*.

**A held-out final month (the last 21 sessions), never touched during
development.** For a fixed target, a twin model is trained without that month
and evaluated on it once (`evaluate-target --holdout --record`). The model
used live is then trained on everything, with the twin's result kept beside it
as the evidence for how far to trust it.

**Metrics for the ranking**, in the order they matter:

- **Realised net gain of the five picks**, after costs: brokerage, STT,
  exchange fees, SEBI fee, stamp duty, GST. stock-analyzer's charge schedule is
  ported to `eval/charges.py`. A round trip costs about 0.075–0.083%, which
  erases most windows under half a percent, so gross performance is close to
  meaningless.
- **Precision@5 and NDCG@5**: did the picks land in the day's genuine top set.
- **Quantile coverage**: do 50% of outcomes fall inside the p25–p75 band. A
  miscalibrated band is worse than no band.
- **Duration accuracy** by bucket.
- **Adverse rate**: the share of picks that fell at least
  `score_order.adverse_move_pct` (default 2%) below their as-of price later in
  the session. Realised gain is the best window still available, so a ranking
  can raise it just by picking more volatile symbols; this shows the cost.

**Metrics for a fixed target**, at every 15-minute point:

- the hit rate of the model's top 5, against every symbol (the base rate),
  the 5 most volatile now, and the 5 widest by 20-day ATR;
- net per trade: buy at the close of that minute, sell at the target, the
  stop, or the end of the window, less costs;
- calibration: the model's probability against how often it came true;
- band coverage: the share of outcomes at or below the predicted p25, p50 and
  p75;
- filters: of the symbols a rule keeps ("chance ≥ 50%", "median ≥ target"),
  how many made it;
- screener evidence (section 4.2).

### 6.1 What the measurements say

| Question | Tested on | Model | Simple screen | Base | Net per trade (model) |
|---|---|---|---|---|---|
| Best window still open, top 5 at T1 | 87 development sessions | — | `highest_volatility` 1.39% net, 22% adverse | random 0.91%, 8% adverse | — |
| Best window, stage-1 model at T1 | 20 sessions after 46 training | 1.41% net | `highest_volatility` 2.43% | — | — |
| +1% within 40 minutes, top 5 | held-out month, 21 sessions | 18.3% | most volatile now 21.4% | 5.5% | −0.06% (screen −0.05%) |
| +0.5% before the close, top 5 | held-out month, 21 sessions | 69% | most volatile now 69% | 55% | +0.03% (every stock +0.03%) |
| +0.5% above the next open, top 5 | held-out month, 21 sessions | 69% | widest range the day before 83% | 73% | −0.49% (screen −0.14%, every stock −0.06%) |

What follows from it:

- **More recent or finer training data has not helped.** For +0.5% before the
  close, models trained on only the last 30 days, or on every minute rather
  than every 3rd, all reached 66–69% for the top 5 on the held-out month, and
  all kept the same too-low band.
- **No model beats the simplest volatility screen yet**, on any of the three
  questions, and none makes money after costs.
- **The fixed-target models rank well above chance**, about three times the
  base rate for +1% in 40 minutes, but that is what volatility alone delivers.
- **Their raw probabilities are not to be read at face value.** Above about
  20%, the +1% model is overconfident: a predicted 40–60% came true 20–23% of
  the time. The +0.5% model's band was too low for a busier month: the actual
  best gain beat its median 60% of the time, not 50%. Answers therefore show
  the calibrated chance.
- **"+0.5% before the close" is the norm, not a signal.** From 10:30, more
  than half of these stocks trade 0.5% above that price at some point before
  the close. A filter on it keeps most of the universe.

---

## 7. Serving

### The latency requirement decides the architecture

A prediction for a moment does not depend on who asks, so **the model never
runs on the request path**. It runs on a schedule or in a background thread;
requests read a cached answer.

```
scheduler ──► feature build ──► score ──► in-memory cache ──► REST
  (T0, T1, every 15 min)                        │
                                                └──► predictions table (scored next day)

live board ──► latest finished minute ──► score every symbol ──► /dashboard
  (every 20 s, recomputed when the minute moves)
```

| Stage | Budget |
|---|---|
| Cache lookup and serialise | < 5 ms p99 |
| Full request, warm | < 10 ms p99 |
| Scheduled recompute | < 5 s, off the request path |
| Live board refresh (62 symbols, with drivers) | ~6 s, off the request path |

A single uvicorn process with the answers held in memory is enough for one
user; nothing here needs Redis until several consumers exist.

### Three ways to read a prediction

- **The scheduled ranking**: `/predictions/top`, recomputed at T0, T1 and
  every 15 minutes, with `as_of` on every response so a consumer can tell a
  fresh answer from a stale one. With no model served, it falls back to the
  `highest_volatility` baseline, and says so in `model_version`.
- **The live board** (`/dashboard`) shows every symbol at the latest finished
  minute:
  - its calibrated chance, its band, and the inputs that moved it most (each
    set to the median stock's value);
  - its screener data, with what each signal has been measured to be worth;
  - a short conclusion built from those figures;
  - beside them: the volatility screen, the held-out evidence, the
    calibration table, and what stock-analyzer is polling, holding and
    requesting.
- **On demand**: `predict` prints every symbol now, and `summary` (or
  `scripts/today.sh`) prints the stocks expected to reach the target before
  the close. Each comes with its median gain and range and its held-out
  track record; the summary also gives the stand-outs and the held-out
  evidence. `scripts/tomorrow.sh` does the same for the next session, after
  scoring today's recorded picks.
- **Kept current daily**: `scripts/refresh.sh` runs after stock-analyzer's
  evening batch has stored the day. It scores the recorded picks, retrains
  every model in use on the new session, redoes the held-out checks weekly
  and prunes superseded models. `scripts/schedule-refresh.sh` installs it in
  launchd for 19:15 on weekdays.

### API

```
GET  /health                 liveness, model version, cache age
GET  /predictions/top?n=5    the ranking, with bands and as_of
GET  /predictions/{symbol}   one symbol's expectation
GET  /model                  version, trained window, held-out metrics
POST /predict/intraday       stock-analyzer's per-candle model contract
GET  /dashboard              the live board; /dashboard/data is its JSON
```

`/predict/intraday` implements the request and response in stock-analyzer's
`docs/MODEL-CONTRACT.md`, so one service can serve both this project's answers
and stock-analyzer's live signals. stock-analyzer calls it for every tracked
symbol on each poll once its `model.enabled` is true. Every poll re-sends the
session so far; each candle's signal is stored once, the first time it is
computed. The service also logs every request per symbol, for the dashboard.

### Every prediction is stored

Written to stock-predictor's own SQLite with its `as_of`, and scored the next
day against what actually happened. Without this, drift is invisible and the
held-out metrics slowly become a historical curiosity.

---

## 8. Project layout

```
stock-predictor/
├── pyproject.toml
├── params.example.toml  candidate-input parameters, all defaults (section 4.1)
├── scripts/            today.sh, tomorrow.sh (section 7), refresh.sh + schedule-refresh.sh (daily)
├── src/stock_predictor/
│   ├── params.py      the parameters: defaults, TOML/JSON file, --set overrides
│   ├── filters.py     hard filter rules over the candidate inputs
│   ├── data/          stock-analyzer's views (live or finished), its screener data, own store, saved frames
│   ├── labels/        ranking targets and adverse moves, with the leakage rule in one place
│   ├── features/      per-symbol, cross-sectional, regime, candidate inputs; parity with the Java contract
│   ├── fundamentals/  FundamentalScore protocol + null and CSV implementations
│   ├── news/          NewsSentiment protocol + null implementation
│   ├── models/        barrier labels, GBDT backend, stage 1, stage 2, calibration, registry, training set
│   ├── eval/          splits, baselines, charges, metrics, backtest, ablation, fixed targets, screener evidence
│   ├── serving/       FastAPI app, scheduler, cache, live board and dashboard, on-demand summary
│   └── cli.py         build-features | train | predict | summary | models | evaluate
│                      | evaluate-target | screener-evidence | backtest | ablate | params | frames
│                      | serve | score
├── tests/
└── docs/              DESIGN.md (this), ARCHITECTURE.md
```

Python 3.9+, pandas, numpy, scikit-learn or LightGBM, FastAPI and uvicorn,
Parquet for saved training frames, pytest. Dependencies deliberately few.

---

## 9. Delivery

| Phase | Deliverable | State |
|---|---|---|
| 1 | Read stock-analyzer's views; build labels; **leakage tests** | Done |
| 2 | **Baselines + evaluation harness** | Done |
| 3 | Feature pipeline, with parity against the Java fixture | Done |
| 4–5 | Stage-1 model, stage-2 ranking, magnitude and duration | Done; does not beat `highest_volatility` |
| 5a | Candidate screening inputs as parameters, plus `ablate` (4.1) | Done; none switched on |
| 5b | Fixed targets, calibration, `evaluate-target` (1.2, 6) | Done; ties the volatility screen |
| 6 | Serving, scheduler, prediction storage, live reads | Done |
| 6a | Live board, on-demand summary, screener evidence (3.4, 7) | Done |
| 7 | Fundamentals score (3.4) | Interface only; needs a point-in-time source |
| 8 | News sentiment (3.5) | Interface only; stretch |

Phases 1 and 2 are the whole project in miniature. The baselines were hard
to beat, and that was known before a serving stack existed.

---

## 10. Risks, stated plainly

**The model may never beat the volatility screen.** Intraday movement is
mostly noise, and costs eat sub-0.5% moves. So far no model has beaten it
(section 6.1). That is a realistic outcome, not a failure of execution, and
the evaluation is built to show it rather than paper over it. The ranking
baseline, the calibrated chances and the accumulated data remain useful either
way.

**A probability read at face value misleads.** A model trained on calm months
and used in a busy one is off in a predictable direction. Every answer shows
the calibrated chance beside the raw one. The calibration and the held-out
check need redoing as the regime moves, which `today.sh --evaluate` does.

**Overfitting on regime features.** Symbol-specific features have ample data;
market-wide ones (breadth, mean return, dispersion) have one value per session
and will happily memorise them. Keep them few, keep the trees shallow, and
prefer the simpler model whenever the difference is within noise.

**Data accumulation is the long pole.** Minute history cannot be backfilled
past the provider's window, so the dataset grows one session per day. A model
should be retrained as sessions finish (`today.sh` does it for the fixed
target) and re-evaluated on a rolling held-out month.

**Costs are not a detail.** Every metric is computed net. A strategy showing 1.2%
gross and 0.1% net is a loss dressed as a win.

**The ground truth can move under the project.** stock-analyzer's detector
changed once (section 3.2), and the default this project reads stopped being
produced. A silent switch would change what every ranking metric means.

---

## 11. Open questions

1. **Which detector should the ranking use?** The default stopped being
   produced on 2026-11-05. Moving to `hold5` restores current data but changes
   what a window is, and the ranking's results would need re-measuring.
2. **Does the universe stay at these 62 symbols?** A fixed universe is easier to
   evaluate and quietly introduces selection bias. A changing one needs
   point-in-time membership to avoid look-ahead.
3. **Which fixed target is worth acting on?** +0.5% before the close happens
   for most stocks most days; +1% in 40 minutes is rare and not beaten by the
   model. The useful question may sit between them, or need a stop.
4. **Which fundamentals source?** It must be point-in-time (published dates)
   to be usable as a feature at all.
5. **What is an acceptable hit rate?** The threshold at which you would act
   should be set before the next metric is seen, not after.
