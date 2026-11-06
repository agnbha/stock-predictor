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

**Which ground truth, and which sessions.** stock-analyzer versions its
detector, and the series moves on. As of 2026-11-05:

| Detector | Symbols | Final sessions | Status |
|---|---|---|---|
| `topk-nonoverlap/highlow/v1` | 49 | 68, ends 2026-09-07 | no longer produced |
| `topk-nonoverlap/highlow/min0.5/v1` | 62 | 108, 2026-06-02 → 2026-11-04 | **current default** |

The `min0.5` detector keeps only windows of at least 0.5%, which is roughly
where costs stop erasing a move (section 6). Staying on a retired detector fails
quietly: newer sessions have no labels and drop out of every metric, and symbols
added since score zero whenever they are picked. Training and evaluation
therefore use only **completed sessions**, meaning those with `final` labels
under the configured detector. That excludes today's session while it runs,
because it has only `live` labels.

**The universe grew from 49 to 62 symbols**, and the newer symbols' history was
backfilled to 2026-08-07. Some were added because they currently top a CAN SLIM
chart. That is selection on recent performance, applied to the past, so it
flatters any strategy that favours them in the backfilled period. Open question
2 (section 11) is now live, not hypothetical.

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
RSI, moving-average distances, 10-day range position and beta from its trend
views.

Every daily value is as of the last completed session *before* the one being
predicted. The 10-day range position is computed from daily candles, not read
from `v_range_context`. That view holds only the latest row per symbol, so it
leaked the future into every historical session. The 52-week range position is
dropped for the same reason. stock-analyzer keeps only a current snapshot of
the 52-week high and low, and its daily history (108 sessions) is too short to
rebuild one point-in-time. Beta is computed from daily candles over the
sessions before the one being predicted, rather than read from
`v_symbol_beta`, which also covers every session to date. Over all sessions
the two agree to 1e-15. Applied to past sessions, the snapshot was well off:
on 2026-07-15 the median symbol's beta was off by 0.26 and the worst by 1.15.
Point-in-time beta is noisy for a symbol with only a few sessions of history.
On 2026-08-17, one symbol added that month had a beta about 10 away from
its current value.

**Every row depends only on candles at or before its own minute.** A training
frame is built once per session, to 15:00, so any value taken at the build
time and copied to every row reaches back into earlier rows. Two did this
until 2026-11-05. The distances from the 5/20/50-day averages used the 15:00
close, so each 09:30 row knew where the stock ended the day. The regime
clock was 345 minutes on every row. Both are now computed per row. A test
builds a session at 10:30 and at 15:00 and requires the shared rows to match
on every column. The saved training frames (section 8) rely on the same
property.

**Cross-sectional** — every per-symbol feature also enters as its **rank and
z-score within that day's universe**. The question is "which of these 49 is
best today", so absolute values matter less than relative standing, and
day-level normalisation strips out the market-wide move that would otherwise
dominate.

**Regime** — market breadth, mean universe return, dispersion, day of week,
minutes since open. These are identical for every symbol on a day and exist to
let the model condition on the kind of day it is.

### 4.1 Candidate screening inputs — kept as parameters, measured before use

Seven inputs were proposed for picking stocks and tested elsewhere first. The
opening check was tested on 5,982 stock-sessions across these 62 symbols. The
other inputs were tested on about 182,000 stock-sessions of the whole liquid
market, against **next-day** moves. Most showed no reliable effect.

They are kept here as named parameters in `params.py`, all defaults in
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
not apply, and the symbol is kept. `stock-predictor ablate` produces the
numbers below. The target is the net-of-cost best window still open after T,
measured at 09:45 so the opening check is known, over the 87 development
sessions to 2026-10-05. Intervals are 95% bootstraps over sessions.

| Input | Parameters (default) | Earlier finding (next day) | Measured here | Status |
|---|---|---|---|---|
| **Opening check** | `check_time` 09:45, `opening_range_minutes` 15, `allowed_range_states` [above], `require_above_vwap` true | Range and VWAP state at 09:45 told nothing about the rest of the day | **Informative, in reverse.** IC −0.15 [−0.19, −0.11]. The 14% above the range and above VWAP have 0.18 pp *less* remaining gain. Below-range, below-VWAP symbols have the most. As a filter it cuts both baselines' top-5 by ~0.4 pp | Off. Never use it as a breakout filter. As a feature it is the one worth a retrain test |
| **Run-up cap** | `lookback_days` 5, `max_pct` 10 | No reliable effect | The cap rarely binds (98% pass). The 2% over it have +1.2 pp larger windows *and* a 32% adverse rate against 8%. That is a volatility signal. As a filter it costs 0.10 pp | Off |
| **RSI limits** | `period` 14, `min` 30, `max` 70 | No reliable effect | IC +0.07 [+0.04, +0.10]: small. No pass/fail difference. As a filter it costs 0.16 pp on `yesterdays_leaders` | Off. `rsi14` is already a feature |
| **Volume** | `lookback_days` 20, `min_relative_volume` 1.0 | No reliable effect | IC +0.01 [−0.02, +0.05]. Nothing | Off |
| **Close location** | `min` 0.5, `max` 1.0 (prior-day close in its range) | No reliable effect | IC +0.04 [+0.01, +0.07]: barely there. No filter effect | Off |
| **Score order** | `rank_by` score, `adverse_move_pct` 2.0 | The strongest fifth by score is just more volatile: 6.6% fall more than 4% next day, against 4.4% for the market | **Confirmed in kind.** By ATR quintile, net gain rises 0.63 → 1.11% while the adverse rate rises 5% → 14% | `score` stays the default. `risk_adjusted` is available. `adverse_rate` is reported for every strategy |
| **Medium-term momentum** | `lookback_days` 120, `max_dist_from_high_pct` 10 | Points the right way over 10–20 sessions, but weakly. CAN SLIM covers that horizon | **Not measurable yet.** Daily history is 108 sessions, short of 120. A 60-day proxy: IC +0.03 [−0.03, +0.08], no top-5 effect | Off. Re-measure once 120 sessions exist: late November 2026 for the original 49 symbols, early 2027 for the 13 added later |

How to read this:

- **The target here is an upper bound.** It is the best window still
  available, not a trade anyone took. Anything that tracks how much a symbol
  moves will look informative against it. The adverse-rate columns are there
  to catch that, which is how the run-up and score-order rows read.
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
None of the seven meets the filter rule today.

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

How that number orders the universe is a parameter (`score_order.rank_by`,
section 4.1). `score` ranks on it directly. `risk_adjusted` divides it by the
symbol's ATR % first, so a symbol does not reach the top merely by being the
most volatile. Any active filters are applied to the ordering before the top
five are taken.

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
- **Adverse rate.** The share of picks that fell at least
  `score_order.adverse_move_pct` (default 2%) below their as-of price later in
  the session. Realised gain is the best window still available, so a ranking
  can raise it just by picking more volatile symbols. This metric shows the
  cost of doing so. On the 87 development sessions at T1, `highest_volatility`
  earns 1.39% net with a 22% adverse rate, against 0.91% and 8% for
  `random_five`.

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
├── params.example.toml  candidate-input parameters, all defaults (section 4.1)
├── src/stock_predictor/
│   ├── params.py      the parameters: defaults, TOML/JSON file, --set overrides
│   ├── filters.py     hard filter rules over the candidate inputs
│   ├── data/          read-only access to stock-analyzer's views; own store; saved training frames
│   ├── labels/        target construction and adverse moves, with the leakage rule in one place
│   ├── features/      per-symbol, cross-sectional, regime, candidate inputs; parity with the Java contract
│   ├── fundamentals/  FundamentalScore protocol + null implementation
│   ├── models/        ranker, magnitude, duration; train and predict
│   ├── eval/          walk-forward splitter, baselines, metrics, ablation of candidate inputs
│   ├── serving/       FastAPI app, scheduler, cache
│   └── cli.py         build-features | train | evaluate | backtest | ablate | params | serve | score
├── tests/
└── docs/DESIGN.md
```

Python 3.12, FastAPI, uvicorn, LightGBM, pandas, Parquet for saved training
frames (one file per completed session), pytest. Dependencies deliberately few.

[ARCHITECTURE.md](ARCHITECTURE.md) has the module diagram, where each piece of
data is stored, and the measured daily churn.

---

## 9. Delivery

| Phase | Deliverable | Why this order |
|---|---|---|
| 1 | Read stock-analyzer's views; build labels; **leakage tests** | Everything rests on the label being honest |
| 2 | **Baselines + evaluation harness** | Without this, no later number means anything |
| 3 | Feature pipeline, with parity against the Java fixture | |
| 4 | Ranker, measured against the baselines | First point a model exists |
| 5 | Magnitude quantiles and duration buckets | Only useful once the ranking is |
| 5a | Candidate screening inputs as parameters, plus `ablate` (4.1) | Each must be measured on this target before it is switched on |
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
