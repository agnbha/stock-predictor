# Stock Predictor

Predicts the best intraday opportunities for the current session — which
symbols, for how long, and for what gain — and serves that ranking over a
low-latency REST API.

A sibling of [`stock-analyzer`](../stock-analyzer), which it reads from and
never writes to. stock-analyzer records what *was* available; this tries to say
what *will be*.

The full design, including what the available data does and does not support,
is in [docs/DESIGN.md](docs/DESIGN.md). This README describes what is built.

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

## Use

```bash
# Phase 1–2 — the honest core: labels + baselines + evaluation
stock-predictor evaluate --point T1 --max-sessions 30
stock-predictor backtest --point T1                 # purged walk-forward folds

# Phase 3–5 — features and the two-stage model
stock-predictor build-features 2026-08-27 --point T1 --out frame.parquet
stock-predictor train                               # fits stage 1 on the dev sessions
stock-predictor evaluate --point T1 --model latest --holdout --record

# Phase 6 — serving
stock-predictor serve --model latest --port 8001
stock-predictor score --today 2026-09-03            # fill realised gains on past predictions
```

`evaluate` prints a table ranked by realised net gain, e.g.:

```
strategy                    realised_net_g  precision_at_k  ndcg_at_k  quantile_cover  duration_accur  hits
highest_volatility                  1.5733          0.2286     0.4165          0.0000          0.0000  4.57
yesterdays_leaders                  1.1978          0.1143     0.3315          0.1000          0.2429  4.29
random_five                         1.1678          0.0571     0.3352          0.0286          0.0000  4.57
sklearn-triple-barrier/2026-08-04   1.1071          0.0857     0.3335          0.2857          0.1143  4.00
```

The `highest_volatility` baseline is the honest competitor (design §5). As the
design says up front, the model may not beat it on a few months of one
exchange's data — the harness is built to detect that early, not paper over it.

### API

```
GET  /health                 liveness, model version, cache age
GET  /predictions/top?n=5    the ranking, with bands and as_of
GET  /predictions/{symbol}   one symbol's expectation
GET  /model                  version, trained window, held-out metrics
POST /predict/intraday       stock-analyzer's existing per-candle model contract
```

Every ranking response carries `as_of` so a consumer can tell a fresh answer
from a stale one. `/predict/intraday` implements the request/response shape in
stock-analyzer's `docs/MODEL-CONTRACT.md`, so one service can serve both this
project's ranking and stock-analyzer's live per-candle signals.

## Layout

```
src/stock_predictor/
├── clock.py         NSE session grid (port of MarketClock)
├── config.py        paths, prediction points, duration buckets
├── data/            read-only access to stock-analyzer's v_* views; own SQLite store
├── labels/          target construction — the leakage rule, in one place
├── features/        parity port of the 26-feature contract, + cross-sectional + regime
├── fundamentals/    FundamentalScore protocol + null / CSV implementations
├── news/            NewsSentiment protocol + null implementation
├── models/          triple-barrier labels, GBDT backend, stage-1 forward model,
│                    stage-2 deterministic aggregation, calibration, registry
├── eval/            charges, purged walk-forward splitter, 4 baselines, metrics, backtest
├── serving/         in-memory cache, recompute scheduler, FastAPI app
└── cli.py           build-features | train | evaluate | backtest | serve | score

tests/               leakage, targets, splits, metrics, charges, baselines,
                     feature parity (vs stock-analyzer's pinned fixture), aggregation, API
```

## Status by phase (design §9)

| Phase | State |
|---|---|
| 1 · labels + leakage tests | **done** — runs against the real DB |
| 2 · baselines + evaluation harness | **done** — 4 baselines, net-of-cost metrics, walk-forward |
| 3 · feature pipeline | **done** — parity port verified to 1e-9 against the Java fixture |
| 4–5 · ranker + magnitude/duration | **done** — stage-1 GBDT, stage-2 reduction, calibration |
| 6 · serving + scheduler + storage | **done** — FastAPI, T0/T1/15-min recompute, prediction store |
| 7 · fundamentals score | **interface only** — no data source exists yet (design §3.3); `NullFundamentalScore` by default, `CsvFundamentalScore` for a bring-your-own screener export |
| 8 · news sentiment | **interface only** — `NullNewsSentiment` (design §3.4, stretch) |

### Deviations from the design, and why

- **GBDT backend.** The design names LightGBM; it needs an OpenMP runtime that
  is not always present. [`models/gbdt.py`](src/stock_predictor/models/gbdt.py)
  prefers LightGBM and falls back to scikit-learn's `HistGradientBoosting*`
  (same model family). `backend_name()` reports which is active; the model
  version string records it.
- **Feature storage.** The design mentions DuckDB. The pipeline currently
  hands back an in-memory pandas frame and `build-features --out` writes
  Parquet/CSV; a DuckDB-backed store is an optional extra, not yet wired as the
  default.
- **`time_of_day_prior` baseline** reads stock-analyzer's `hot_window` table.
  If that table is empty in your checkout, the baseline returns nothing (shown
  as zeros) — populate it on the stock-analyzer side to activate it.
- **Held-out month.** `CONFIG.holdout_sessions` (default 21) is set aside by
  `holdout_split`; `--holdout` is opt-in on every command so development never
  touches it by accident.

## Tests

```bash
pytest -q          # 37 tests; those needing stock-analyzer's DB skip cleanly without it
```
