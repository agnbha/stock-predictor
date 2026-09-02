# Stock Predictor

Predicts the best intraday opportunities for the current session — which
symbols, for how long, and for what gain — and serves that ranking over a
low-latency REST API.

A sibling of [`stock-analyzer`](../stock-analyzer), which it reads from and
never writes to. stock-analyzer records what *was* available; this tries to say
what *will be*.

**Status: design only.** Nothing is implemented yet. The full design, including
what the available data does and does not support, is in
[docs/DESIGN.md](docs/DESIGN.md).

## The short version

- **Target** — the five best windows across the universe for the current
  session, as three separate predictions: a ranking of symbols, a gain
  distribution (p25/p50/p75, not a point estimate), and a duration bucket.
- **Timing** — predictions are made as of a moment T, and may only use data
  available at T while scoring only windows that open after it. That leakage
  rule is the project's central correctness constraint.
- **Latency** — the model never runs on the request path. It runs on a
  schedule; requests read a cached answer, target p99 under 10 ms.
- **Honesty** — every metric is net of costs, evaluation is day-level
  walk-forward against baselines built first, and a held-out month is never
  touched during development.

## Three constraints worth knowing before starting

1. **52 weeks of minute data does not exist.** The provider caps intraday
   history at roughly three months; 66 sessions are currently stored. A full
   year of daily candles is available for slower features.
2. **Train per candle, not per day.** There are 1.17M minute candles (~41k
   non-overlapping 30-minute episodes) but only 66 sessions. The market factor
   explains just 16.9% of daily variance, so symbols are far from
   interchangeable — the data is abundant for symbol-specific signal and scarce
   only for market-regime features.
3. **Requirement 3.3 has no data source.** Filings and revenue data are not
   available from the current API and need a separate provider decision.
