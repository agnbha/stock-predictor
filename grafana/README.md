# Dashboards

Two dashboards in a "Stock Predictor" folder, on their own Grafana at
http://localhost:3001, beside stock-analyzer's on 3000. Grafana reads two
SQLite files in place, through the
[SQLite datasource plugin](https://grafana.com/grafana/plugins/frser-sqlite-datasource/):

- **StockPredictor**: this project's store, `data/stock-predictor.db`. It holds
  what `today.sh` and `tomorrow.sh` recorded at every run, how each prediction
  did once its session closed, and the held-out checks.
- **StockAnalyzer (read-only)**: stock-analyzer's database, for the actual
  minute candles.

Both are opened with `query_only`, so a dashboard can never change either file.

## Run it

```bash
./grafana/run-local.sh      # http://localhost:3001, admin/admin
```

It reuses the Grafana build and SQLite plugin that stock-analyzer's
`grafana/run-local.sh` downloaded, when they exist. Otherwise it downloads
its own into `grafana/.runtime/`, about 100 MB, once. Its settings and logs
live in `grafana/.runtime/`; nothing of stock-analyzer's is changed. Delete
`grafana/.runtime/` to undo everything. Ctrl-C stops it.

The dashboards fill as runs are recorded. `scripts/schedule-today.sh install`
records a run every 10 minutes from 09:20 to 14:50 on weekdays. The evening
refresh scores each day's predictions.

## Picks, per stock

One row of charts for each stock picked in the **Stocks** box. The box lists
the stocks by chance at the latest run, most likely first.

- **Range in price, entry and chance at each run.** Time runs along the
  bottom, with one candle per `today.sh` run:
  - **Candle**: the *Range in price*, from the p25 to the p75 of the best
    price expected later today.
  - **Orange line**: the *Entry price*, the most to pay. It is 0.6% below the
    range's low end, or the price at the run when that is lower.
  - **Bars** (right axis, 0–100%): the *chance* of the target at that run.
  - **Dashed line**: the price at the run.
  - The tooltip also shows the *edge*: the chance minus the base rate, in
    points.
- **What the price did.** stock-analyzer's minute candles over the same time.
  Did the price reach the range?
- **Green markers** on both charts are runs where the stock was flagged, at
  10 or more points above the base rate.

Above the rows are:
- the latest run's base rate, and how many stocks were flagged or had 50%+;
- every stock at the latest run, with the same columns as `today.sh`;
- the flagged count and the base rate, run by run.

The **Model** box switches between `today.sh`'s models. Pick
`...-stop0.5pct-close` for `today.sh --stop 0.5`.

## How the picks did

These panels use the scored predictions in the time range:
- how often each group reached the target, session by session: flagged
  stocks, those with 50%+, each run's top 5, and every stock;
- the chance given against what happened (are the chances honest?);
- the edge against what happened;
- a track record per stock.

Below them is the **held-out check**: the month a twin of the model was
tested on without having seen it. It shows the top 5 against the most
volatile stocks and against any stock, the record by edge that the flag
quotes, and the base rate through the day.

Held-out checks are copied into the store's `model_evaluation` table by
`evaluate-target --record`; `today.sh --evaluate` and the Friday refresh run
it. To copy the ones recorded before the table existed, run
`stock-predictor models --to-store`.

## Changing a dashboard

Edit `grafana/make_dashboards.py` and run it:

```bash
.venv/bin/python grafana/make_dashboards.py
```

Grafana picks the new files up within seconds. `tests/test_grafana.py` runs
every query against the real schemas, and checks that the committed JSON is
what the script writes. A dashboard changed in Grafana's editor is kept until
the script is next run, which overwrites it, so carry the change over into
the script.
