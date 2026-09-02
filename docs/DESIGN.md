# Stock Predictor — Project Design

A Python service that predicts the best intraday opportunities for the current
session — which symbols, for how long, and for what gain — and serves that
ranking over a low-latency REST API.

It is a sibling of `stock-analyzer`, which it reads from and never writes to.
stock-analyzer records what *was* available; stock-predictor tries to say what
*will be*.

---

## 0. Read this part first

Three facts about the available data shape everything below, and pretending
otherwise would produce a design that looks impressive and does not work.

**The 52 weeks of minute data in requirement 3.1 does not exist.** The provider
caps intraday history at roughly three months. stock-analyzer currently holds
**66 sessions** of minute candles (2026-06-02 to 2026-09-02, 1.17M candles, 49
symbols). A full year of *daily* candles is available (248 sessions) and is
worth using for slower features, but minute-level history cannot be bought back.
The design therefore trains on what exists and is built to accumulate forward.

**There is far more training data than there are days — provided the model is
trained at the right granularity.** An earlier draft of this design claimed the
effective sample was 66 because symbols on a day share one market. Measured, that
is wrong: the market factor explains only **16.9%** of daily return variance
across this universe, and mean cross-sectional dispersion is **0.96%** against a
mean market move of **0.47%**. Symbol-specific movement is roughly twice the
market-wide movement, so symbols are much closer to independent than that
argument assumed.

The data available depends entirely on what a training example *is*:

| Training unit | Examples available |
|---|---|
| Minute candle, forward-looking label | **1,168,059** |
| Non-overlapping 30-minute episodes | **40,908** |
| Symbol-session | **3,227** |
| Session (market regime only) | **66** |

Only the last row is genuinely scarce, and it constrains only features that are
identical for every symbol on a day — breadth, market return, dispersion. Those
have 66 observations and should be used sparingly. Everything symbol-specific
has orders of magnitude more.

The design consequence is section 5: the model is trained per candle, where the
data is, and its output is *aggregated* into the daily top-five. The granularity
of the answer does not have to be the granularity of the training.

**Requirement 3.3 has no data source.** There is no filings or revenue data
anywhere in stock-analyzer, and the Groww API does not serve financial
statements. A quarterly score out of 100 needs NSE/BSE XBRL filings or a paid
vendor, and parsing XBRL is a project in itself. It is phased accordingly.

None of this makes the project not worth building. It does mean the honest
first deliverable is **a rigorous way to tell whether a model beats a simple
baseline**, and only then a model.

---

## 1. What is predicted

stock-analyzer already computes, for every symbol and session, the top three
non-overlapping windows in which a gain was available — entry time, exit time,
gain percent, duration. That table is the ground truth this project learns from,
and it is already versioned by detector so a rules change produces a comparable
parallel series rather than silently rewriting history.

The target is the **five best windows across the whole universe for the current
session**, ranked by gain. Concretely, a prediction is five rows of:

```
symbol · expected entry window · expected duration · expected gain %, with a range
```

That is deliberately three separate predictions with very different difficulty,
and treating them as one regression would hide that:

| Component | Method | Why |
|---|---|---|
| **Which symbols** | Learning-to-rank over the day's universe | Relative ordering is far more learnable than absolute outcome |
| **Gain %** | Quantile regression (p25 / p50 / p75) | A point estimate of a noisy quantity invites false confidence; a band is honest |
| **Duration** | Classifier over buckets (<15m, 15–60m, 60–180m, >180m) | Exact minutes are not predictable; the bucket is what changes a decision |

Current label statistics, for calibration of expectations: the mean top-ranked
window gains **1.61%** and lasts **140 minutes**.

---

## 2. Timing, and the leakage rule

This is where projects like this usually go wrong, so it is settled before
anything else.

A prediction is always made **as of a moment T**, and:

- features may use only data available at or before T;
- the label counts only windows whose **entry is strictly after T**.

Predicting at 09:30 that a window which opened at 09:20 was good is not a
prediction. It is reading the answer. Because the label depends on T, the
training set is built per prediction point, not per day.

Three prediction points, each a distinct model context:

| Point | When | What it can see | Honest expectation |
|---|---|---|---|
| **T0** | 09:00, pre-open | Prior sessions, overnight gap proxies, fundamentals | Weakest. Mostly a volatility ranking |
| **T1** | 09:30, after the opening range | T0 plus the day's open, first 15 minutes, gap, opening-range break | The most useful single call of the day |
| **Tn** | every 15 min to 15:00 | Everything so far that session | Narrowing, as fewer minutes remain to be captured |

The remaining-opportunity framing has a pleasant property: the label naturally
shrinks through the day, so the model learns that a 2% window is less likely at
14:30 than at 09:30 without being told.

---

## 3. Data

### 3.1 Minute candles and derived curves — from stock-analyzer

Read-only, over its published views rather than its base tables:
`v_intraday_candles`, `v_daily_candles`, `v_symbol_trend`, `v_support_resistance`,
`v_market_breadth`, `v_range_context`, `v_opportunity_markers`. Those views are
a deliberate contract; the tables behind them are not.

Coupling is one-way and read-only. stock-predictor never writes to that
database — it keeps its own.

### 3.2 The day's own shape

Open, high, low so far, opening-range break, gap versus prior close, VWAP and
distance from it, volume against the same minute on other days, position within
the day's range, first-hour high. stock-analyzer already computes most of this
and pins it with a shared fixture (`feature-parity.json`); reusing that contract
avoids two implementations drifting apart.

### 3.3 Fundamentals score, 0–100 per quarter

No source exists yet. The design keeps it behind an interface with a null
implementation so everything else can be built and evaluated without it:

```python
class FundamentalScore(Protocol):
    def score(self, symbol: str, as_of: date) -> float | None: ...
```

Candidate sources, in order of effort: a screener-style data provider (fastest),
NSE/BSE XBRL filings (free, slow to parse), a paid vendor (fastest and dullest).
The scoring itself — revenue and margin trend, debt trajectory, cash conversion,
surprise versus consensus — is straightforward once the data lands.

**A caution worth stating:** quarterly fundamentals move four times a year and
intraday windows resolve in hours. Expect this feature to shift the *ranking
between symbols* slightly and to say almost nothing about *today*. It should be
built for that modest role, not treated as the missing ingredient.

### 3.4 News — stretch

Deliberately last. Headline sentiment is the feature most likely to leak
(publication timestamps are unreliable), most likely to be stale, and hardest to
evaluate. Not started until 3.1–3.3 are measurably working.

---

## 4. Features

Three groups, and the middle one is what makes a ranker work:

**Per-symbol** — the 26 features already defined in stock-analyzer's contract
(returns over several windows, volatility, ATR, VWAP distance, volume z-score
against the same time of day, opening-range distances, gap, event flags), plus
RSI, moving-average distances, 52-week range position and beta from its trend
views.

**Cross-sectional** — every per-symbol feature also enters as its **rank and
z-score within that day's universe**. The question is "which of these 49 is
best today", so absolute values matter less than relative standing, and
day-level normalisation strips out the market-wide move that would otherwise
dominate.

**Regime** — market breadth, mean universe return, dispersion, day of week,
minutes since open. These are identical for every symbol on a day and exist to
let the model condition on the kind of day it is.

---

## 5. Models

Two stages, and the split between them is the most important structural decision
in the project.

### Stage 1 — a per-candle model, trained where the data is

```
forward     LightGBM, one row per (symbol, minute)
            label: triple-barrier outcome over the next H minutes
            ~1.17M rows, ~41k non-overlapping episodes
```

For each minute, what happens over the next H minutes: an upper barrier at
+u%, a lower at −d%, a time barrier at H, with the barriers scaled by recent ATR
so they mean the same thing in calm and volatile conditions. This is the same
labelling stock-analyzer's model contract already describes, so the two share a
definition rather than inventing separate ones.

This stage sees over a million examples. It is where any real signal will be
found, and it is trained without ever knowing what a "top five" is.

### Stage 2 — aggregate into the answer

The daily ranking is mostly a deterministic reduction of stage 1, not a second
learned model:

```
for each symbol, at prediction point T:
    expected best window  = max over remaining minutes of the forward model's
                            predicted move, discounted by its own probability
    rank the universe by that number, take five
```

Magnitude and duration come from the same stage-1 outputs — quantiles of the
predicted move for the gain band, and the horizon at which the upper barrier is
most often struck for the duration bucket.

Only a thin calibration sits on top: mapping stage-1 scores onto realistic
gain expectations, a handful of parameters. That is the one place the 66-session
limit genuinely binds, and fitting a few parameters on 66 days is defensible
where fitting a full ranker on them would not be.

Gradient-boosted trees throughout: tabular features, and nothing here calls for
a neural network.

### Baselines that must be beaten

Built and measured **before** any model, because without them a metric is just a
number:

1. **Yesterday's leaders** — the five that had the biggest windows last session.
2. **Highest recent volatility** — the five with the widest 20-day ATR. This is
   the one to watch; volatility genuinely predicts the *size* of a move and is
   the honest competitor.
3. **stock-analyzer's time-of-day prior** — already implemented there.
4. **Random five** — the floor.

---

## 6. Evaluation

**Split by day, never within one.** All 49 symbols of a session go to the same
fold. Purged walk-forward with an embargo of one session either side of each
boundary, since labels look forward within a day.

Worth separating two things that are easy to conflate — the earlier draft of
this design conflated them. Splitting at day granularity is about **leakage**:
overlapping label horizons mean two minutes from the same session are not
independent enough to sit on opposite sides of a split. It says nothing about
how many training examples exist. There are 66 folds' worth of *boundaries* and
over a million *examples*.

**Metrics**, in the order they matter:

- **Realized net gain of the five picks**, after costs — brokerage, STT,
  exchange fees, stamp duty, GST, and slippage. stock-analyzer's `ChargeModel`
  already encodes these. A round trip costs roughly 0.05–0.1%, which erases most
  windows under half a percent, so gross performance is close to meaningless.
- **Precision@5 and NDCG@5** — did the picks land in the day's genuine top set.
- **Quantile coverage** — do 50% of outcomes fall inside the p25–p75 band. A
  miscalibrated band is worse than no band.
- **Duration accuracy** by bucket.

**A held-out final month, never touched during development.** With 66 sessions
this costs about a third of the data, and it is still the right call: it is the
only defence against having tuned your way to a number.

---

## 7. Serving

### The latency requirement decides the architecture

A prediction for a session does not depend on who asks or when, so **the model
never runs on the request path**. It runs on a schedule; requests read a cached
answer.

```
scheduler ──► feature build ──► score ──► in-memory cache ──► REST
  (T0, T1, every 15 min)                        │
                                                └──► predictions table (for later scoring)
```

| Stage | Budget |
|---|---|
| Cache lookup and serialise | < 5 ms p99 |
| Full request, warm | < 10 ms p99 |
| Scheduled recompute (49 symbols) | < 5 s, off the request path |

A single uvicorn process with the ranking held in a dict is enough for one user;
nothing here needs Redis until several consumers exist.

### API

```
GET /health                        liveness, model version, cache age
GET /predictions/top?n=5           the ranking, with bands
GET /predictions/{symbol}          one symbol's expectation
GET /model                         version, trained window, held-out metrics
POST /predict/intraday             stock-analyzer's existing model contract
```

`/predictions/top` returns, for each pick: symbol, expected entry window,
duration bucket, gain quantiles, model version, and **`as_of`** — the moment the
prediction was computed. A consumer must be able to tell a fresh answer from a
stale one.

The last endpoint matters for cohesion: stock-analyzer already defines a
per-candle signal contract in `docs/MODEL-CONTRACT.md` with a feature-parity
fixture, and its `RestIntradaySignalModel` points at `model.service.url`.
Implementing that endpoint here means one service serves both its ranking and
stock-analyzer's live signals.

### Every prediction is stored

Written to stock-predictor's own SQLite with its `as_of`, and scored the next
day against what actually happened. Without this, drift is invisible and the
held-out metrics slowly become a historical curiosity.

---

## 8. Project layout

```
stock-predictor/
├── pyproject.toml
├── src/stock_predictor/
│   ├── data/          read-only access to stock-analyzer's views; own store
│   ├── labels/        target construction, with the leakage rule enforced in one place
│   ├── features/      per-symbol, cross-sectional, regime; parity with the Java contract
│   ├── fundamentals/  FundamentalScore protocol + null implementation
│   ├── models/        ranker, magnitude, duration; train and predict
│   ├── eval/          walk-forward splitter, baselines, metrics
│   ├── serving/       FastAPI app, scheduler, cache
│   └── cli.py         build-features | train | evaluate | serve | backtest
├── tests/
└── docs/DESIGN.md
```

Python 3.12, FastAPI, uvicorn, LightGBM, pandas, DuckDB for feature storage,
pytest. Dependencies deliberately few.

---

## 9. Delivery

| Phase | Deliverable | Why this order |
|---|---|---|
| 1 | Read stock-analyzer's views; build labels; **leakage tests** | Everything rests on the label being honest |
| 2 | **Baselines + evaluation harness** | Without this, no later number means anything |
| 3 | Feature pipeline, with parity against the Java fixture | |
| 4 | Ranker, measured against the baselines | First point a model exists |
| 5 | Magnitude quantiles and duration buckets | Only useful once the ranking is |
| 6 | FastAPI serving, scheduler, prediction storage | |
| 7 | Fundamentals score (3.3) | Needs a data source decision first |
| 8 | News sentiment (3.4) | Stretch, and only if 1–7 are working |

Phases 1 and 2 are the whole project in miniature. If the baselines turn out to
be hard to beat, that is worth knowing in week one rather than after building a
serving stack.

---

## 10. Risks, stated plainly

**The model may not beat the volatility baseline.** Intraday movement is mostly
noise, and costs eat sub-0.5% moves. This is a
realistic outcome, not a failure of execution, and the evaluation is designed to
detect it early rather than paper over it. If it happens, the useful products
are still the ranking baseline itself and the accumulated labelled data.

**Overfitting on regime features.** Symbol-specific features have ample data;
market-wide ones (breadth, mean return, dispersion) have 66 observations and
will happily memorise them. Keep them few, keep the trees shallow, and prefer
the simpler model whenever the difference is within noise.

**Data accumulation is the long pole.** Minute history cannot be backfilled
past the provider's window, so the dataset grows one session per day. A model
trained today should be retrained monthly and re-evaluated on a rolling
held-out period.

**Costs are not a detail.** Every metric is computed net. A strategy showing 1.2%
gross and 0.1% net is a loss dressed as a win.

**Fundamentals may not help intraday.** Built as a small ranking adjustment, and
dropped if it does not earn its place on the held-out set.

## 11. Open questions

1. **Which fundamentals source?** This decides whether phase 7 is a week or a
   month, and modelling before deciding risks building to a shape the source
   will not fit.
2. **Does the universe stay at these 50 symbols?** A fixed universe is easier to
   evaluate and quietly introduces selection bias; a changing one needs
   point-in-time membership to avoid look-ahead.
3. **Is a pre-open (T0) prediction wanted at all**, or is T1 after the opening
   range enough? T0 is the weakest of the three and the most work to keep honest.
4. **What is an acceptable hit rate?** Five picks a day at 1.6% mean available
   gain is a meaningful edge if even two land — but the threshold at which you
   would actually act should be set before the first metric is seen, not after.
