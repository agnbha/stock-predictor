# Stock Predictor

Predicts the best intraday opportunities for the current session — which
symbols, for how long, and for what gain — and serves that ranking over a
low-latency REST API.

A sibling of [`stock-analyzer`](../stock-analyzer), which it reads from and
never writes to. stock-analyzer records what *was* available; this tries to say
what *will be*.

The full design, including what the available data does and does not support,
is in [docs/DESIGN.md](docs/DESIGN.md). This README describes what is built.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the module diagram, storage
locations and daily churn.

## What it does

Given a moment **T** in a trading session, it produces the five best windows
across the whole universe that are still open after T, as three separate
predictions:

- **which symbols** — a ranking of the day's universe;
- **gain %** — a p25 / p50 / p75 band, not a point estimate;
- **duration** — a bucket (`<15m`, `15–60m`, `60–180m`, `>180m`).

The model never runs on the request path. A scheduler recomputes the answer at
T0 (09:00), T1 (09:30) and every 15 minutes to 15:00; requests read a cached
ranking. Every prediction is stored with its `as_of` and scored the next day
against what actually happened.

### The two rules the code is built around

1. **The leakage rule (design §2).** A prediction made as of T may use only
   data at or before T, and its label counts only windows whose entry is
   *strictly after* T. This lives in exactly one place —
   [`labels/targets.py`](src/stock_predictor/labels/targets.py) — and
   `leakage_audit()` re-checks it over real sessions in
   [`tests/test_leakage.py`](tests/test_leakage.py).
2. **Everything is net of costs (design §6).** Indian equity intraday
   brokerage + STT + exchange + SEBI + stamp duty + GST is a large fraction of
   a small move. [`eval/charges.py`](src/stock_predictor/eval/charges.py) ports
   stock-analyzer's charge schedule; a round trip costs ≈0.075%, which erases
   most sub-0.5% windows.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or: .[models,serving,dev] as needed
```

The core (data access, labels, baselines, evaluation) needs only `numpy` and
`pandas`. `[models]` adds the GBDT stack, `[serving]` adds FastAPI, `[store]`
adds DuckDB/pyarrow, `[dev]` adds pytest.

stock-analyzer's SQLite database is expected at
`../stock-analyzer/data/stock-analyzer.db`. Override with `STOCK_ANALYZER_DB`.
Ground truth comes from its `topk-nonoverlap/highlow/min0.5/v1` detector
(override with `STOCK_PREDICTOR_DETECTOR`). Training and evaluation use only
*completed* sessions, meaning those with final labels under that detector.

## Use

```bash
# Phase 1–2 — the honest core: labels + baselines + evaluation
stock-predictor evaluate --point T1 --max-sessions 30
stock-predictor backtest --point T1                 # purged walk-forward folds

# Phase 3–5 — features and the two-stage model
stock-predictor build-features 2026-08-27 --point T1 --out frame.parquet
stock-predictor train                               # fits stage 1 on the dev sessions
stock-predictor train --minute-step 5               # every 5th minute: a fifth of the rows
stock-predictor frames                              # saved training frames and their disk use
stock-predictor evaluate --point T1 --model latest --holdout --record

# Phase 5a — candidate screening inputs (design §4.1)
stock-predictor params                              # effective parameters, as TOML
stock-predictor ablate                              # measure every input at 09:45
stock-predictor --set rsi.filter=true evaluate --at 09:45
stock-predictor --params my-params.toml train       # train with chosen inputs as features

# A fixed target: P(the price touches +1% within the next 40 minutes)
stock-predictor train --target 1 --horizon 40 --minute-step 3       # development sessions
stock-predictor evaluate-target --holdout --model <that version> --record   # top 5 vs simple screens
stock-predictor screener-evidence --model <that version> --record   # is the screener data worth anything?
stock-predictor train --target 1 --horizon 40 --minute-step 3 --all-sessions   # to use live
stock-predictor predict                                             # every symbol, right now

# +0.5% at any point before today's close, with the best gain still available
scripts/today.sh                                                    # the summary, whenever you want it
scripts/tomorrow.sh                                                 # how today's picks did, then the next session's
stock-predictor scorecard --days 30                                 # each stock's record from recorded picks
scripts/refresh.sh                                                  # the daily refresh (after stock-analyzer's 18:30 batch)
scripts/schedule-refresh.sh install                                 # run it every weekday at 19:15 (launchd)
stock-predictor models                                              # what each model was trained on
stock-predictor train --target 0.5 --to-close --band max-gain --minute-step 3 --all-sessions
stock-predictor predict --model latest --min-chance 50               # more likely than not

# Phase 6 — serving
stock-predictor serve --model latest --port 8001
stock-predictor score --today 2026-09-03            # fill realised gains on past predictions
```

`evaluate` prints a table ranked by realised net gain. Over the 87
development sessions at T1 (abridged):

```
strategy              realised_net_gain  precision_at_k  adverse_rate
highest_volatility               1.3942          0.1977        0.2172
yesterdays_leaders               1.3808          0.1908        0.1667
time_of_day_prior                0.9951          0.1172        0.1276
random_five                      0.9092          0.0920        0.0828
```

The `highest_volatility` baseline is the honest competitor (design §5). Its
adverse rate shows what it pays for that gain: more than a fifth of its picks
fall 2% or more after T. As the design says up front, the model may not beat it on a
few months of one exchange's data. The harness is built to detect that early,
not paper over it.

### Candidate screening inputs (design §4.1)

Seven inputs proposed for picking stocks are kept as parameters in
[`params.py`](src/stock_predictor/params.py), with every default in
[`params.example.toml`](params.example.toml):
- the opening check at 09:45
- run-up cap
- RSI limits
- volume
- prior-day close location
- score order
- nearness to the N-day high

Each has a `feature` switch (a model column) and a `filter` switch (drop
symbols before ranking). Both are **off by default**: tested elsewhere on
next-day moves, most showed no reliable effect. `stock-predictor ablate`
measures each one on this project's target before anything is switched on.
Design §4.1 has the current numbers.

### A fixed target: +1% within 40 minutes

`train --target 1 --horizon 40` trains stage 1 on a plain question asked of
every minute: from this candle's close, does the price touch +1% within the
next 40 minutes? (`--stop 0.5` would require it before a 0.5% fall.)
`predict` then answers it for every symbol in the session in progress, from
stock-analyzer's live candles, in about 8 s. It shows the model's probability
next to how often that probability came true on the 20 sessions held back
from a calibration fit. It also shows each symbol's current volatility rank,
because that ranking is the one to beat.

Measured on the held-out month (21 sessions, 2026-10-07..11-05), with a model
trained on the 88 sessions before it, every 15 minutes from 09:30 to 14:45:

| Top 5 at each point, by | Touched +1% within 40 min | Net per trade |
|---|---|---|
| — (every symbol: the base rate) | 5.5% | −0.08% |
| the model's P(target) | 18.3% | −0.06% |
| volatility right now (14-minute ATR as % of price) | **21.4%** | −0.05% |
| widest 20-day ATR (`highest_volatility`) | 20.0% | −0.07% |

Net per trade: buy at the close, sell at +1% or after 40 minutes, less
0.083% costs.

The model finds stocks about 3.3 times as likely as average to make the move,
but no better than ranking by current volatility, which needs no model.
Nothing in the table makes money after costs. Its probabilities are too high
above ~20%: on sessions it had not trained on, a predicted 40–60% came true
about 20–23% of the time. `predict` therefore shows the calibrated figure too.
The 20 development sessions before the held-out month gave the same ordering
(18.0% vs 21.9%).

### +0.5% before the close

`train --target 0.5 --to-close --band max-gain` asks, of every minute: will
the price trade 0.5% above this close at any point before 15:29? Its band is
the best gain still available that day: the highest later price against the
price now, as p25 / median / p75. Fixed targets train on every finished
session, because their labels come from the candles, not from
stock-analyzer's detector.

In the held-out month (21 sessions, 2026-10-09..11-09), checked every 15
minutes from 09:30 to 15:00:

| | Reached +0.5% before the close |
|---|---|
| Any stock | 55% (74% at 09:30, 59% at 10:30, 27% at 15:00) |
| The model's top 5 | 69% |
| The 5 most volatile now | 69% |
| Symbols the model gave 50% or more | 62.5% (about 22 a point) |

A 0.5% gain at some point in the day is the norm for these stocks, not a
signal, so the filter keeps most of the universe. The model ranks only
slightly better than chance. Its band was too low for that busier month: the
actual best gain beat the predicted median 60% of the time, not 50%. Buying
the top 5, selling at +0.5% or at the close, made +0.03% a trade after
0.083% costs, the same as buying every stock.

`scripts/today.sh` prints this as a summary, on demand. It covers:
- the stocks with a 50% or better chance;
- each stock's median expected gain and range (the top 15 in full, the rest
  grouped by median);
- the stand-outs;
- the held-out evidence on how far to trust it.

It retrains the model first when a session has finished since it was trained
(about 2 minutes); otherwise it takes about 7 seconds. Options:
- `--target 1`: a different gain;
- `--min-chance 60`: a stricter cut;
- `--top 20`: list more stocks in full;
- `--at 11:00`: an earlier minute;
- `--session 2026-11-09`: a past day, marked in-sample when the model trained
  on it;
- `--json FILE`: also write the summary as JSON;
- `--retrain` or `--no-retrain`: force or skip retraining;
- `--evaluate`: redo the held-out check.

It wraps `stock-predictor summary`, which takes the same options except
`--retrain`, `--no-retrain` and `--evaluate`.

**What the models are trained on.** The models `today.sh` and `tomorrow.sh`
use train on every finished session stock-analyzer has. `today.sh`'s model
uses every minute of each session, about 2.5 million rows; `tomorrow.sh`'s
uses every stock-session. The daily refresh retrains both after each new
session. The only model that leaves anything out is the twin behind "how much
to trust it" and the "Last month" column. It is trained on everything except
the latest 21 sessions, because those are what it is tested on.
`stock-predictor models` lists what every model was trained on.

Training on only the latest month was tested and is not used. `train
--recent-days N` remains in the CLI for experiments; the scripts never pass
it. On the same held-out month, every variant was trained without it:

| Trained on | Rows | Top 5 reached +0.5% | Net per trade |
|---|---|---|---|
| Full history, every 3rd minute | 573k | 68.9% | +0.02% |
| Full history, every minute | 1.71M | 68.4% | +0.01% |
| Last 30 days, every minute | 427k | 66.4% | +0.04% |
| Last 30 days, every 3rd minute | 143k | 66.4% | +0.02% |
| (the 5 most volatile / any stock) | | 69.5% / 54.5% | −0.02% / +0.03% |

Neither the most recent month nor every minute changes the result; the
differences are within noise. The band stays too low in all four (the best
gain beat the median 59–61% of the time). The held-out month was busier than
the month before it too, so a recent window did not catch up with it. The
scripts train on the full history, at every minute.

### Tomorrow, the scorecard and the daily refresh

**`scripts/tomorrow.sh`** (`today.sh --tomorrow`) answers the same question
for the next session: will each stock trade 0.5% above that session's open
before its close, and how far could it go? There are no candles for a session
before it opens, so this model (`train --next-session`) has one row per stock
and session. Its inputs are the stock's state at 15:00 the session before;
its label is the next session's high against its open. The next session's
date comes from stock-analyzer's NSE holiday list.

On the held-out month (21 sessions, a model trained without them):

| Top 5 each session, by | Reached +0.5% above the open | Net per trade |
|---|---|---|
| every stock | 73% | −0.06% |
| the model | 69% | −0.49% |
| the widest range the session before | **83%** | −0.14% |
| the widest 20-day ATR | 74% | −0.41% |

Three in four stocks get there anyway. The model's picks did worse than a
random stock. The widest-range screen found the movers but lost money when
they missed, so `tomorrow.sh` lists that screen's five as well.

**Track record.** Each pick shows how the model did on that stock in the
held-out month. "14/18 76%" means it said 50%+ on 18 occasions, the stock got
there on 14, and its call, yes or no, was right 76% of the time.

**The scorecard.** Both scripts record every run's predictions for every stock
(`summary --record`; `--no-record` to skip). `tomorrow.sh` opens with how
today's picks did: for each `today.sh` run, how many picks reached +0.5% from
the price then, its top 5, and every stock for comparison. It also shows how
yesterday's `tomorrow.sh` picks did from today's open. Picks are scored once
the session has closed; before that, the scorecard reports progress so far.
`stock-predictor scorecard --days 30` gives each stock's record from recorded
picks.

**The daily refresh.** `scripts/refresh.sh` keeps all of this current. It
waits until stock-analyzer has stored the day's candles, which its evening
batch does at 18:30. Then it:
1. scores the day's recorded picks;
2. retrains `today.sh`'s and `tomorrow.sh`'s models, and the last-month ones
   if you use them, building the new session's saved frame on the way;
3. on Fridays, redoes the held-out checks;
4. deletes superseded models, keeping three of each kind.

Each step does nothing if it is already done, and the log is
`data/logs/refresh.log`. `scripts/schedule-refresh.sh install` runs it every
weekday at 19:15 with launchd, as stock-analyzer schedules its own jobs.
`status` and `uninstall` check and remove it. A run missed while the machine
slept starts when it wakes.

### The dashboard

`stock-predictor serve` also serves a live board at
<http://127.0.0.1:8001/dashboard>, refreshed every 15 s. It uses the
fixed-target model trained through the latest session (`--live-model` to
choose another). For every symbol, at the latest minute 90% of symbols have
finished, it shows:

- **Likelihood.** The model's probability, and how often that probability
  came true on sessions it had not trained on, with the number of cases
  behind that rate. The model's top five are drawn in blue.
- **What moved it.** The inputs that moved this symbol's probability most.
  Each input, with its rank and z-score across stocks, is set to today's
  median stock and the probability recomputed; the change is shown in points.
- **Screener data from stock-analyzer.** Fundamentals from screener.in, the
  latest CANSLIM letters, RS rating and signal, the bullish score and
  watchlist. Where a signal has history, the board also shows what it has
  been worth on this target.
- **A conclusion.** A few sentences built only from the figures above.

Beside the table:
- how the numbers are made: the model, its training, the held-out comparison
  and the calibration table;
- the five most volatile stocks now, the screen the model is measured against;
- the scheduler's ranking;
- stock-analyzer's activity:
  - the symbols its monitor polls and how fresh each one is;
  - what it has bought and holds today;
  - every request it has made to `POST /predict/intraday`, per symbol.

That last list stays empty while stock-analyzer has `model.enabled=false`;
it then scores with its own time-of-day prior, and the board says so.

The board needs a fixed-target model. Its held-out and screener panels fill
from the `--record` runs above, which are kept beside the model trained
without the held-out month and found by target. After 14:49, a 40-minute
window would run past the close, so the board holds 14:49.

`#open=WELCORP` in the address opens that symbol's details, and
`?theme=dark` picks the theme.

Screener evidence, for touching +1% within 40 minutes. Each group is compared
with the rest, using only what was published before each session's open:

| Signal | Sessions | In group | Rest | Among the model's top 5 (in / rest) |
|---|---|---|---|---|
| Bullish score in the top fifth (previous session) | 89 | 5.2% | 3.0% | 16% / 19% |
| On the watchlist made after the previous session | 89 | 7.1% | 3.6% | 18% / 18% |
| CANSLIM RS rating 80+ (run before the open) | 8 | 7.1% | 4.0% | 21% / 9% |
| 4+ of 7 CANSLIM letters passed | 8 | 9.1% | 4.0% | 26% / 9% |

On its own, a bullish score or a watchlist place has meant roughly twice the
chance. Among the model's top picks it has made no difference, which
suggests the model already sees what those signals see. CANSLIM looks strong
there too, but its runs began on 2026-10-26: eight sessions, and too few
symbol-days to read either way.

The fundamentals (P/E, ROE, growth) are one snapshot per company, so they
cannot be tested on past sessions without look-ahead. The board shows them,
labelled as not measured.

### Saved training frames

Building one session's feature frame from stock-analyzer's views takes about
7 s. `train` saves each completed session's frame as Parquet under
`data/frames/` (`STOCK_PREDICTOR_FRAMES`) and reads it back next time in about
0.04 s; missing frames are built in `--jobs` processes. Frames hold features
only. Labels for the target being trained are attached on load, so a new
target or horizon reuses them. `evaluate --model`, `evaluate-target` and
`ablate --model` cut the same frames back to their prediction point instead of
rebuilding. A frame is rebuilt automatically when the code that builds it
changes, when stock-analyzer rewrites that session or an earlier one, or when a
different set of candidate features is switched on (each set has its own
folder). `--no-frame-cache` builds everything and saves nothing. A session
costs about 7 MB on disk; delete the folder to reclaim it.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the measurements.

### API

```
GET  /health                 liveness, model version, cache age
GET  /predictions/top?n=5    the ranking, with bands and as_of
GET  /predictions/{symbol}   one symbol's expectation
GET  /model                  version, trained window, held-out metrics
POST /predict/intraday       stock-analyzer's existing per-candle model contract
GET  /dashboard              the live board (above); /dashboard/data is its JSON
```

Every ranking response carries `as_of` so a consumer can tell a fresh answer
from a stale one. `/predict/intraday` implements the request/response shape in
stock-analyzer's `docs/MODEL-CONTRACT.md`, so one service can serve both this
project's ranking and stock-analyzer's live per-candle signals.

## Layout

```
src/stock_predictor/
├── clock.py         NSE session grid (port of MarketClock)
├── config.py        paths, detector version, prediction points, duration buckets
├── params.py        candidate-input parameters: defaults, TOML/JSON file, --set
├── filters.py       hard filter rules over the candidate inputs
├── data/            read-only access to stock-analyzer's v_* views (and, for display and
│                    evidence only, its screener data and monitor activity); own SQLite
│                    store; saved training frames (Parquet, one per session)
├── labels/          target construction and adverse moves — the leakage rule, in one place
├── features/        parity port of the 26-feature contract, + cross-sectional + regime
│                    + candidate inputs (point-in-time)
├── fundamentals/    FundamentalScore protocol + null / CSV implementations
├── news/            NewsSentiment protocol + null implementation
├── models/          triple-barrier labels (ATR-scaled or a fixed target), GBDT backend,
│                    stage-1 forward model, stage-2 deterministic aggregation,
│                    calibration, registry, training set (saved frames, labels attached
│                    on load, column-major matrix, minute step)
├── eval/            charges, purged walk-forward splitter, 4 baselines, metrics, backtest,
│                    ablation of candidate inputs, fixed-target hit rates, screener evidence
├── serving/         in-memory cache, recompute scheduler, FastAPI app, live board and
│                    dashboard page, log of stock-analyzer's requests
└── cli.py           build-features | train | predict | summary | evaluate | evaluate-target
                     | screener-evidence | backtest | ablate | params | frames | serve | score
scripts/today.sh     the on-demand summary: retrains when stale, records, then `summary`
scripts/tomorrow.sh  the next session's summary, after today's scorecard (today.sh --tomorrow)
scripts/refresh.sh   the daily refresh; schedule-refresh.sh installs it in launchd

tests/               leakage, targets, splits, metrics, charges, baselines,
                     feature parity (vs stock-analyzer's pinned fixture), aggregation, API,
                     candidate inputs (params, point-in-time values, filters, ablation),
                     storage (signals written once, saved frames, rows independent of build time),
                     fixed targets (labels, clock-minute windows, live session, versions),
                     dashboard (drivers, screener point-in-time, request log, endpoints, board)
```

## Status by phase (design §9)

| Phase | State |
|---|---|
| 1 · labels + leakage tests | **done** — runs against the real DB |
| 2 · baselines + evaluation harness | **done** — 4 baselines, net-of-cost metrics, walk-forward |
| 3 · feature pipeline | **done** — parity port verified to 1e-9 against the Java fixture |
| 4–5 · ranker + magnitude/duration | **done** — stage-1 GBDT, stage-2 reduction, calibration |
| 5a · candidate screening inputs | **done** — 7 inputs as parameters, off by default; `ablate` measures them |
| 5b · fixed targets | **done** — +X% within H minutes or before the close, calibrated chance, best-gain band, `evaluate-target`; ties the volatility screen |
| 6 · serving + scheduler + storage | **done** — FastAPI, T0/T1/15-min recompute, prediction store, live reads of the session in progress |
| 6a · live board + summary | **done** — `/dashboard`, `summary` / `scripts/today.sh`, screener evidence |
| 7 · fundamentals score | **interface only** — stock-analyzer's screener data is a snapshot, not point-in-time (design §3.4); `NullFundamentalScore` by default, `CsvFundamentalScore` for a bring-your-own export |
| 8 · news sentiment | **interface only** — `NullNewsSentiment` (design §3.5, stretch) |

### Deviations from the design, and why

- **GBDT backend.** The design names LightGBM; it needs an OpenMP runtime that
  is not always present. [`models/gbdt.py`](src/stock_predictor/models/gbdt.py)
  prefers LightGBM and falls back to scikit-learn's `HistGradientBoosting*`
  (same model family). `backend_name()` reports which is active; the model
  version string records it.
- **Feature storage.** The design mentions DuckDB. Training frames are saved
  as Parquet instead, one file per completed session (above). A file per
  session needs no schema or server, and is all training and evaluation need:
  they read whole sessions, never query across them.
- **`time_of_day_prior` baseline** reads stock-analyzer's `hot_window` table.
  If that table is empty in your checkout, the baseline returns nothing (shown
  as zeros) — populate it on the stock-analyzer side to activate it.
- **52-week range position** is dropped from the features. stock-analyzer has
  only a current snapshot of it, so using it on past sessions would be
  look-ahead. The 10-day range position and beta are now computed
  point-in-time from daily candles.
- **Held-out month.** `CONFIG.holdout_sessions` (default 21) is set aside by
  `holdout_split`; `--holdout` is opt-in on every command so development never
  touches it by accident.

## Tests

```bash
pytest -q          # 98 tests; those needing stock-analyzer's DB skip cleanly without it
```
