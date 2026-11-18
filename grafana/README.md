# Dashboards

Two dashboards in a **"Stock Predictor" folder in stock-analyzer's Grafana**
(http://localhost:3000), beside its "Stock Analyzer" folder. There is no
second Grafana. The dashboards read two SQLite files in place, through the
[SQLite datasource plugin](https://grafana.com/grafana/plugins/frser-sqlite-datasource/):

- **StockPredictor**: this project's store, `data/stock-predictor.db`. It holds
  what `today.sh` and `tomorrow.sh` recorded at every run, how each prediction
  did once its session closed, and the held-out checks.
- **StockAnalyzer (read-only)**: stock-analyzer's database, for the actual
  minute candles. This is separate from stock-analyzer's own datasource, so
  every Stock Predictor panel reads without writing.

Both are opened with `query_only`, so a dashboard can never change either
file. Neither is the default; stock-analyzer's stays the default.

## Add them

Start stock-analyzer's Grafana once with its `grafana/run-local.sh`, then:

```bash
grafana/add-to-stock-analyzer.sh                             # then restart stock-analyzer's Grafana
GRAFANA_PASSWORD=<its admin password> grafana/add-to-stock-analyzer.sh   # or: the running one reloads
grafana/add-to-stock-analyzer.sh status                      # are they there?
grafana/add-to-stock-analyzer.sh remove                      # take them out again
```

stock-analyzer's `run-local.sh` keeps Grafana's provisioning in its
`grafana/.runtime/provisioning/`, and on each start it rewrites only its own
two files there. This script adds two more, both named `stock-predictor.yaml`:
one for the datasources, and one for the folder, which points at this repo's
`grafana/dashboards`. Nothing of stock-analyzer's own changes. If its
`grafana/.runtime/` is ever deleted, run the script again.

The dashboards fill as runs are recorded. `scripts/schedule-today.sh install`
records a run every 10 minutes from 09:20 to 14:50 on weekdays. The evening
refresh scores each day's predictions.

## Predicted Picks

One row of charts for each stock picked in the **Stocks** box. The box lists
the stocks by chance at the latest run, most likely first. You can also pick
stocks from the **Latest run, every stock** table: click a stock's name and
choose **Add it to the charts below** (keeps the stocks already picked) or
**Show only it**. Either fills the Stocks box, keeping the time range and the
model. To take a stock off, untick it in the Stocks box.

The time picker has the same options as stock-analyzer's **Candles and Calls**
page, and it opens on **Today's trading hours** (09:15 today to now).

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
  Did the price reach the range? As on stock-analyzer's Candles and Calls,
  hovering shows Open, Close, High, Low and Volume, then:
  - **EMA 10m** (yellow) and **EMA 60m** (blue): exponential moving averages
    of the minute closes. Each restarts at its session's first close and
    appears once its 10 or 60 minutes have passed.
  - **Previous day's close** (dotted): the close of the session before.

  Candles and Calls itself draws simple averages (SMA).
- **Green markers** on both charts are runs where the stock was flagged, at
  10 or more points above the base rate.

Above the rows are:
- the latest run's base rate, and how many stocks were flagged or had 50%+;
- every stock at the latest run, with the same columns as `today.sh` (click a
  name to chart it);
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
