"""Writes grafana/dashboards/*.json, the Stock Predictor dashboards.

    .venv/bin/python grafana/make_dashboards.py

Edit here and re-run, rather than editing the JSON: the queries, units and
layout stay consistent. (A dashboard changed in Grafana can be exported over
its file instead; this script would then overwrite it on the next run.)

Two datasources, both SQLite files read in place (provisioning/datasources):
``stock-predictor-sqlite``, this project's store (what today.sh and
tomorrow.sh recorded, how it did, the held-out checks), and
``stock-analyzer-readonly``, stock-analyzer's database, read without writing
(the actual minute candles).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("dashboards")
SP = {"type": "frser-sqlite-datasource", "uid": "stock-predictor-sqlite"}
SA = {"type": "frser-sqlite-datasource", "uid": "stock-analyzer-readonly"}
ENTRY_MARGIN = 0.6  # serving/summary.py ENTRY_MARGIN_PCT
# Prices as stock-analyzer's Candles and Calls shows them: the full value in rupees and paise
# (11,053.15), never abbreviated (Grafana's currency units turn that into ₹11.1K).
PRICE = {"unit": "locale", "decimals": 2}
IN_RANGE = "as_of BETWEEN $__from / 1000 AND $__to / 1000"
# The first dashboard keeps its uid, so links and bookmarks made before it was renamed still open it.
PICKS_UID, PICKS_SLUG = "stockpredictor-picks", "predicted-picks"
# The time picker's options, exactly as stock-analyzer's Candles and Calls dashboard has them
# (grafana/dashboards/02-candles-and-calls.json there). The first is Predicted Picks' default.
QUICK_RANGES = [
    {"display": "Today's trading hours", "from": "now/d+9h+15m", "to": "now"},
    {"display": "This week (Monday 9:15 → now)", "from": "now/w+9h+15m", "to": "now"},
    {"display": "Last 2 days", "from": "now-2d", "to": "now"},
    {"display": "Last 7 days", "from": "now-7d", "to": "now"},
    {"display": "Last 30 days", "from": "now-30d", "to": "now"},
    {"display": "Last 90 days", "from": "now-90d", "to": "now"},
    {"display": "Last 6 months", "from": "now-6M", "to": "now"},
    {"display": "Last 1 year", "from": "now-1y", "to": "now"},
    {"display": "Last 2 years", "from": "now-2y", "to": "now"},
    {"display": "Last 5 years", "from": "now-5y", "to": "now"},
    {"display": "Yesterday", "from": "now-1d/d", "to": "now-1d/d"},
    {"display": "Day before yesterday", "from": "now-2d/d", "to": "now-2d/d"},
    {"display": "This day last week", "from": "now-7d/d", "to": "now-7d/d"},
    {"display": "Previous week", "from": "now-1w/w", "to": "now-1w/w"},
    {"display": "Previous month", "from": "now-1M/M", "to": "now-1M/M"},
    {"display": "Previous fiscal quarter", "from": "now-1Q/fQ", "to": "now-1Q/fQ"},
    {"display": "Previous year", "from": "now-1y/y", "to": "now-1y/y"},
    {"display": "Previous fiscal year", "from": "now-1y/fy", "to": "now-1y/fy"},
    {"display": "Today", "from": "now/d", "to": "now/d"},
    {"display": "Today so far", "from": "now/d", "to": "now"},
    {"display": "This week", "from": "now/w", "to": "now/w"},
    {"display": "This week so far", "from": "now/w", "to": "now"},
    {"display": "This month", "from": "now/M", "to": "now/M"},
    {"display": "This month so far", "from": "now/M", "to": "now"},
    {"display": "This year", "from": "now/y", "to": "now/y"},
    {"display": "This year so far", "from": "now/y", "to": "now"},
    {"display": "This fiscal quarter so far", "from": "now/fQ", "to": "now"},
    {"display": "This fiscal quarter", "from": "now/fQ", "to": "now/fQ"},
    {"display": "This fiscal year so far", "from": "now/fy", "to": "now"},
    {"display": "This fiscal year", "from": "now/fy", "to": "now/fy"},
]
TRADING_HOURS = QUICK_RANGES[0]  # "Today's trading hours": 09:15 today to now
FAMILY = "substr(model_version, 1, instr(model_version, '/') - 1)"
# The sessions (IST dates) the time range touches.
SESSION_FROM = "date($__from / 1000, 'unixepoch', '+5 hours', '+30 minutes')"
SESSION_TO = "date($__to / 1000, 'unixepoch', '+5 hours', '+30 minutes')"


def target(sql: str, ds=SP, time=True, ref="A") -> dict:
    sql = sql.strip()
    return {"datasource": ds, "refId": ref, "queryText": sql, "rawQueryText": sql,
            "queryType": "table", "timeColumns": ["time"] if time else []}


class Layout:
    def __init__(self):
        self.next_id, self.panels = 1, []

    def add(self, panel: dict, x: int, y: int, w: int, h: int) -> dict:
        panel = {"id": self.next_id, "gridPos": {"x": x, "y": y, "w": w, "h": h}, **panel}
        self.next_id += 1
        self.panels.append(panel)
        return panel


def stat(title, sql, unit="none", desc="", decimals=None, color="blue", ds=SP):
    defaults = {"unit": unit, "color": {"mode": "fixed", "fixedColor": color}}
    if decimals is not None:
        defaults["decimals"] = decimals
    return {"type": "stat", "title": title, "description": desc, "datasource": ds,
            "targets": [target(sql, ds, time=False)],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "value", "graphMode": "none", "textMode": "value", "justifyMode": "center"}}


def override(name: str, **props) -> dict:
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": k.replace("__", "."), "value": v} for k, v in props.items()]}


def hidden(name: str) -> dict:  # kept in the tooltip, left out of the plot and the legend
    return override(name, custom__hideFrom={"viz": True, "legend": True, "tooltip": False})


def variable(name, label, sql, multi=False, desc="", refresh=2, ds=SP):
    return {"name": name, "label": label, "type": "query", "datasource": ds, "query": sql, "definition": sql,
            "refresh": refresh, "sort": 0, "includeAll": False, "multi": multi, "description": desc,
            "current": {"selected": False, "text": "", "value": ""}}


def dashboard(uid, title, desc, panels, templating, annotations=(), time_from="now/d", time_to="now/d",
              refresh="1m", quick_ranges=None):
    return {"uid": uid, "title": title, "description": desc, "tags": ["stock-predictor"],
            "timezone": "Asia/Kolkata", "editable": True, "graphTooltip": 1, "schemaVersion": 39, "version": 1,
            "refresh": refresh, "time": {"from": time_from, "to": time_to},
            **({"timepicker": {"quick_ranges": quick_ranges}, "weekStart": "monday"} if quick_ranges else {}),
            "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                                      "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
                                      "name": "Annotations & Alerts", "type": "dashboard"}, *annotations]},
            "templating": {"list": templating}, "panels": panels}


# -- 1. Predicted Picks -----------------------------------------------------------

def predicted_picks() -> dict:
    lay = Layout()
    today_family = f"kind = 'today' AND model_version LIKE '${{family}}/%'"
    latest = (f"(SELECT MAX(as_of) FROM summary_pick WHERE {today_family} AND {IN_RANGE})")
    templating = [
        variable("family", "Model",
                 f"SELECT DISTINCT {FAMILY} FROM summary_pick WHERE kind = 'today' ORDER BY 1",
                 desc="Which today.sh model's runs to show: sklearn-target-0.5pct-close is the default "
                      "(+0.5% at some point before the close); ...-stop0.5pct-close is today.sh --stop 0.5 "
                      "(+0.5% before it falls 0.5%)."),
        variable("symbol", "Stocks",
                 f"SELECT symbol FROM summary_pick WHERE {today_family} AND as_of = {latest} "
                 "ORDER BY chance DESC",
                 multi=True,
                 desc="One chart per stock, listed by chance at the latest run in the time range (the first "
                      "is the most likely). Tick several to compare them."),
    ]
    annotations = [{
        "datasource": SP, "enable": True, "iconColor": "#73BF69", "name": "Flagged",
        "target": target(f"""
SELECT as_of AS time,
       symbol || ' flagged: chance ' || printf('%.0f%%', chance * 100)
         || ', edge ' || printf('%+.0f', (chance - base_rate) * 100) || ' points' AS text,
       symbol AS tags
FROM summary_pick
WHERE {today_family} AND flagged = 1 AND symbol IN (${{symbol:sqlstring}}) AND {IN_RANGE}""")}]

    y = 0
    lay.add(stat("Latest run", f"SELECT MAX(as_of) * 1000 AS \"Latest run\" FROM summary_pick "
                 f"WHERE {today_family} AND {IN_RANGE}", unit="dateTimeAsLocal",
                 desc="The minute of the latest recorded today.sh run in the time range (schedule-today.sh runs "
                      "it every 10 minutes, 09:20 to 14:50)."), 0, y, 6, 4)
    lay.add(stat("Base rate now", f"SELECT MAX(base_rate) * 100 AS \"Base rate\" FROM summary_pick "
                 f"WHERE {today_family} AND as_of = {latest}", unit="percent", decimals=0,
                 desc="How often any stock reached the target from this time of day in the held-out month. A "
                      "chance near it says nothing."), 6, y, 6, 4)
    lay.add(stat("Flagged now", f"SELECT COALESCE(SUM(flagged), 0) AS \"Flagged\" FROM summary_pick "
                 f"WHERE {today_family} AND as_of = {latest}", color="green",
                 desc="Stocks whose chance is 10+ points above the base rate at the latest run."), 12, y, 6, 4)
    lay.add(stat("50%+ now", f"SELECT COALESCE(SUM(passed), 0) AS \"50%+\" FROM summary_pick "
                 f"WHERE {today_family} AND as_of = {latest}",
                 desc="Stocks with a chance of 50% or more at the latest run."), 18, y, 6, 4)
    y += 4

    # Clicking a stock's name reopens this dashboard with the Stocks box set: the stock added to the
    # ones picked, or on its own. Same time range and model.
    here = f"/d/{PICKS_UID}/{PICKS_SLUG}?${{__url_time_range}}&${{family:queryparam}}"
    pick_links = [
        {"title": "Add ${__value.raw} to the charts below", "targetBlank": False,
         "url": here + "&${symbol:queryparam}&var-symbol=${__value.raw:percentencode}"},
        {"title": "Show only ${__value.raw}", "targetBlank": False,
         "url": here + "&var-symbol=${__value.raw:percentencode}"},
    ]
    table = lay.add({
        "type": "table", "title": "Latest run, every stock (click a stock to chart it)", "datasource": SP,
        "description": "Every stock at the latest run, most likely first. **Click a stock's name** to add it to "
                       "the charts below (the Stocks box at the top), or to show only it. Edge is the chance minus "
                       "the base rate, in points; ✓ marks a stock 10+ points above it. Range in price is the "
                       f"p25-p75 of the best price expected later today; Entry is the most to pay, {ENTRY_MARGIN:g}% "
                       "below its low end (or the price then).",
        "targets": [target(f"""
SELECT rank AS "#", symbol AS "Stock", price AS "Price", chance * 100 AS "Chance",
       CASE WHEN base_rate IS NULL THEN NULL ELSE (chance - base_rate) * 100 END AS "Edge",
       CASE WHEN flagged = 1 THEN '✓' ELSE '' END AS "Flag",
       gain_p50 AS "Median gain",
       price * (1 + gain_p25 / 100.0) AS "Range low", price * (1 + gain_p75 / 100.0) AS "Range high",
       MIN(price, price * (1 + gain_p25 / 100.0) * (1 - {ENTRY_MARGIN} / 100.0)) AS "Entry"
FROM summary_pick WHERE {today_family} AND as_of = {latest}
ORDER BY rank""", time=False)],
        "fieldConfig": {"defaults": {"custom": {"align": "auto", "filterable": True}, "decimals": 2},
                        "overrides": [
                            override("#", decimals=0, custom__width=50),
                            override("Chance", unit="percent", decimals=0,
                                     custom__cellOptions={"type": "gauge", "mode": "basic"}, min=0, max=100),
                            override("Edge", decimals=0, custom__cellOptions={"type": "color-text"},
                                     thresholds={"mode": "absolute", "steps": [
                                         {"color": "text", "value": None}, {"color": "green", "value": 10}]}),
                            override("Median gain", unit="percent"),
                            *(override(c, **PRICE) for c in ("Price", "Range low", "Range high", "Entry")),
                            override("Flag", custom__width=60),
                            override("Stock", links=pick_links),
                        ]},
        "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}},
    }, 0, y, 15, 12)
    lay.add({
        "type": "timeseries", "title": "Flagged stocks and the base rate, run by run", "datasource": SP,
        "description": "At each run: how many stocks were flagged (10+ points above the base rate) and how many "
                       "had 50%+, and the base rate itself. The base rate falls through the day, as less of it is "
                       "left.",
        "targets": [target(f"""
SELECT as_of AS time, SUM(flagged) AS "Flagged", SUM(passed) AS "50%+", MAX(base_rate) * 100 AS "Base rate"
FROM summary_pick WHERE {today_family} AND {IN_RANGE}
GROUP BY as_of ORDER BY as_of""")],
        "fieldConfig": {"defaults": {"custom": {"drawStyle": "line", "lineWidth": 2, "pointSize": 5,
                                                "showPoints": "always", "spanNulls": False}},
                        "overrides": [override("Base rate", unit="percent", min=0, max=100,
                                               custom__axisPlacement="right",
                                               color={"mode": "fixed", "fixedColor": "purple"}),
                                      override("Flagged", color={"mode": "fixed", "fixedColor": "green"},
                                               custom__drawStyle="bars", custom__fillOpacity=50),
                                      override("50%+", color={"mode": "fixed", "fixedColor": "blue"})]},
        "options": {"legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
    }, 15, y, 9, 12)
    y += 12

    # One row per stock picked in the Stocks box: the prediction at each run, and what the price did.
    lay.add({"type": "row", "title": "$symbol", "collapsed": False, "repeat": "symbol", "panels": []}, 0, y, 24, 1)
    y += 1
    lay.add({
        "type": "candlestick", "title": "$symbol: Range in price, entry and chance at each run", "datasource": SP,
        "description": "One candle per today.sh run (time along the bottom). The candle spans the **Range in price**: "
                       "the p25 to p75 of the best price expected later today. The orange line is the **Entry price** "
                       f"(the most to pay: {ENTRY_MARGIN:g}% below the range's low end, or the price then). The bars, "
                       "on the right axis, are the **chance** of the target at that run. The dashed line is the price "
                       "at the run. Green markers: runs where the stock was flagged.",
        "targets": [target(f"""
SELECT as_of AS time,
       price * (1 + gain_p25 / 100.0) AS open,
       price * (1 + gain_p75 / 100.0) AS high,
       price * (1 + gain_p25 / 100.0) AS low,
       price * (1 + gain_p75 / 100.0) AS close,
       MIN(price, price * (1 + gain_p25 / 100.0) * (1 - {ENTRY_MARGIN} / 100.0)) AS "Entry price",
       chance * 100 AS "Chance",
       price AS "Price then",
       CASE WHEN base_rate IS NULL THEN NULL ELSE (chance - base_rate) * 100 END AS "Edge"
FROM summary_pick
WHERE {today_family} AND symbol = '${{symbol}}' AND {IN_RANGE}
ORDER BY as_of""")],
        "fieldConfig": {"defaults": {**PRICE,
                                     "custom": {"drawStyle": "line", "lineWidth": 1, "fillOpacity": 0,
                                                "axisPlacement": "left", "axisLabel": "Price"}},
                        "overrides": [
                            override("open", displayName="Range low (p25)"),
                            override("close", displayName="Range high (p75)"),
                            override("high", custom__hideFrom={"viz": False, "legend": True, "tooltip": True}),
                            override("low", custom__hideFrom={"viz": False, "legend": True, "tooltip": True}),
                            override("Entry price", custom__lineWidth=2, custom__showPoints="always",
                                     color={"mode": "fixed", "fixedColor": "orange"}),
                            override("Chance", unit="percent", decimals=0, min=0, max=100,
                                     custom__drawStyle="bars", custom__fillOpacity=35, custom__lineWidth=1,
                                     custom__axisPlacement="right", custom__axisLabel="Chance",
                                     color={"mode": "fixed", "fixedColor": "blue"}),
                            override("Price then", custom__lineStyle={"fill": "dash", "dash": [4, 4]},
                                     color={"mode": "fixed", "fixedColor": "text"}),
                            {**hidden("Edge"), "properties": hidden("Edge")["properties"]
                             + [{"id": "unit", "value": "none"}, {"id": "decimals", "value": 0},
                                {"id": "displayName", "value": "Edge (points)"}]},
                        ]},
        "options": {"mode": "candles", "candleStyle": "candles", "colorStrategy": "open-close",
                    "colors": {"up": "green", "down": "red"}, "includeAllFields": True,
                    "fields": {"open": "open", "high": "high", "low": "low", "close": "close"},
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
    }, 0, y, 14, 11)
    lay.add({
        "type": "candlestick", "title": "$symbol: what the price did (minute candles)", "datasource": SA,
        "description": "stock-analyzer's one-minute candles for the same time range, read without writing to its "
                       "database; the candle still forming is left out. Compare with the predicted range on the "
                       "left: did the price get there? Green markers: runs where the stock was flagged.\n\n"
                       "The tooltip, as on stock-analyzer's Candles and Calls: Open, Close, High, Low, Volume, then\n"
                       "- **EMA 10m / 60m** (yellow, blue): exponential moving averages of the minute closes, "
                       "weighted 2/(N+1) (N = 10, 60). Each restarts with its session's first close and appears "
                       "once N minutes have passed. (Candles and Calls draws simple averages, SMA.)\n"
                       "- **Previous day's close** (dotted): the close of the session before, the price the day's "
                       "move is measured from (stock-analyzer's daily candle).",
        "targets": [target(f"""
-- Whole sessions, so each average starts at its session's open even when the time range starts later.
WITH m AS MATERIALIZED (
  SELECT ts_epoch, session_date, open, high, low, close, volume,
         ROW_NUMBER() OVER (PARTITION BY session_date ORDER BY ts_epoch) AS n
  FROM v_intraday_merged
  WHERE symbol = '${{symbol}}' AND COALESCE(source, '') != 'forming'
    AND session_date BETWEEN {SESSION_FROM} AND {SESSION_TO}
),
-- EMA: the last value moved 2/(N+1) of the way to each new close, from the session's first close.
ema(session_date, n, e10, e60) AS (
  SELECT session_date, n, close, close FROM m WHERE n = 1
  UNION ALL
  SELECT m.session_date, m.n, e.e10 + (2.0 / 11) * (m.close - e.e10), e.e60 + (2.0 / 61) * (m.close - e.e60)
  FROM ema e JOIN m ON m.session_date = e.session_date AND m.n = e.n + 1
),
prev AS (
  SELECT s.session_date,
         (SELECT d.close FROM v_daily_candles d WHERE d.symbol = '${{symbol}}' AND d.session_date < s.session_date
          ORDER BY d.session_date DESC LIMIT 1) AS close
  FROM (SELECT DISTINCT session_date FROM m) s
)
-- The tooltip lists the series in this order.
SELECT m.ts_epoch AS time,
       m.open AS "Open", m.close AS "Close", m.high AS "High", m.low AS "Low", m.volume AS "Volume",
       CASE WHEN m.n >= 10 THEN e.e10 END AS "EMA 10m",
       CASE WHEN m.n >= 60 THEN e.e60 END AS "EMA 60m",
       -- blank on each session's first candle but the first, so the line breaks between sessions
       CASE WHEN m.n > 1 OR m.ts_epoch = (SELECT MIN(ts_epoch) FROM m WHERE ts_epoch >= $__from / 1000)
            THEN p.close END AS "Previous day's close"
FROM m JOIN ema e ON e.session_date = m.session_date AND e.n = m.n
LEFT JOIN prev p ON p.session_date = m.session_date
WHERE m.ts_epoch BETWEEN $__from / 1000 AND $__to / 1000
ORDER BY m.ts_epoch""", ds=SA)],
        "fieldConfig": {"defaults": {**PRICE, "custom": {"lineWidth": 1}},
                        "overrides": [
                            override("Volume", unit="short", decimals=0, custom__axisPlacement="right",
                                     custom__axisLabel="Volume"),
                            override("EMA 10m", custom__drawStyle="line", custom__lineWidth=2,
                                     custom__showPoints="never", color={"mode": "fixed", "fixedColor": "yellow"}),
                            override("EMA 60m", custom__drawStyle="line", custom__lineWidth=2,
                                     custom__showPoints="never", color={"mode": "fixed", "fixedColor": "blue"}),
                            override("Previous day's close", custom__drawStyle="line", custom__lineWidth=2,
                                     custom__lineStyle={"fill": "dash", "dash": [2, 4]}, custom__showPoints="never",
                                     color={"mode": "fixed", "fixedColor": "text"}),
                        ]},
        "options": {"mode": "candles+volume", "candleStyle": "candles", "colorStrategy": "open-close",
                    "colors": {"up": "green", "down": "red", "flat": "gray"}, "includeAllFields": True,
                    "fields": {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"},
                    "tooltip": {"mode": "multi", "sort": "none", "hideZeros": False},
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom", "calcs": []}},
    }, 14, y, 10, 11)

    return dashboard(PICKS_UID, "Predicted Picks",
                     "What today.sh said about each stock at every run (the Range in price, the entry price and the "
                     "chance), and what the price did. Runs come from summary --record, which today.sh and "
                     "schedule-today.sh do by default.",
                     lay.panels, templating, annotations, time_from=TRADING_HOURS["from"],
                     time_to=TRADING_HOURS["to"], quick_ranges=QUICK_RANGES)


# -- 2. How the picks did ----------------------------------------------------------

def how_they_did() -> dict:
    lay = Layout()
    fam = "model_version LIKE '${family}/%'"
    scored = f"{fam} AND scored_at IS NOT NULL AND session_date BETWEEN date($__from / 1000, 'unixepoch', '+5 hours', '+30 minutes') AND date($__to / 1000, 'unixepoch', '+5 hours', '+30 minutes')"
    templating = [variable(
        "family", "Model", f"SELECT DISTINCT {FAMILY} FROM summary_pick ORDER BY 1",
        desc="Which model's recorded picks and held-out check to show. sklearn-nextday-... is tomorrow.sh.")]
    y = 0
    lay.add({"type": "text", "title": "", "options": {"mode": "markdown", "content":
             "**How the recorded picks did.** A pick is scored once its session has closed (the evening refresh, "
             "or `stock-predictor scorecard`). *Reached* means the price got to the target after the run (for "
             "tomorrow.sh, above the next open; with a stop, before the stop). The **held-out check** below is "
             "the month a twin of the model was tested on without having seen it."}}, 0, y, 24, 3)
    y += 3
    lay.add({
        "type": "barchart", "title": "Reached the target, by session", "datasource": SP,
        "description": "For each scored session, across every recorded run: the share that reached the target among "
                       "flagged stocks, stocks with 50%+, each run's top 5, and every stock.",
        "targets": [target(f"""
SELECT session_date AS "Session",
       100.0 * SUM(CASE WHEN flagged = 1 THEN hit END) / NULLIF(SUM(flagged = 1), 0) AS "Flagged",
       100.0 * SUM(CASE WHEN passed = 1 THEN hit END) / NULLIF(SUM(passed = 1), 0) AS "50%+",
       100.0 * SUM(CASE WHEN rank <= 5 THEN hit END) / NULLIF(SUM(rank <= 5), 0) AS "Top 5",
       100.0 * AVG(hit) AS "Every stock"
FROM summary_pick WHERE {scored}
GROUP BY session_date ORDER BY session_date""", time=False)],
        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "decimals": 0,
                                     "custom": {"fillOpacity": 70}},
                        "overrides": [override("Flagged", color={"mode": "fixed", "fixedColor": "green"}),
                                      override("50%+", color={"mode": "fixed", "fixedColor": "blue"}),
                                      override("Top 5", color={"mode": "fixed", "fixedColor": "orange"}),
                                      override("Every stock", color={"mode": "fixed", "fixedColor": "text"})]},
        "options": {"xField": "Session", "orientation": "vertical", "groupWidth": 0.7, "barWidth": 0.9,
                    "showValue": "auto", "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
    }, 0, y, 14, 9)
    lay.add({
        "type": "table", "title": "Sessions", "datasource": SP,
        "description": "Runs, scored stock-predictions, flagged ones and how many reached the target.",
        "targets": [target(f"""
SELECT session_date AS "Session", COUNT(DISTINCT as_of) AS "Runs", COUNT(*) AS "Scored",
       SUM(flagged = 1) AS "Flagged", SUM(CASE WHEN flagged = 1 THEN hit ELSE 0 END) AS "Flagged reached",
       100.0 * AVG(hit) AS "Every stock %", 100.0 * AVG(base_rate) AS "Base rate %"
FROM summary_pick WHERE {scored}
GROUP BY session_date ORDER BY session_date DESC""", time=False)],
        "fieldConfig": {"defaults": {"decimals": 0}, "overrides": []},
        "options": {"showHeader": True, "cellHeight": "sm"},
    }, 14, y, 10, 9)
    y += 9
    lay.add({
        "type": "barchart", "title": "Chance against what happened", "datasource": SP,
        "description": "Scored predictions grouped by the chance given: how often they reached the target. A model "
                       "whose chances can be taken at face value has its bars near the diagonal (a 60-70% group "
                       "reaching it about 65% of the time).",
        "targets": [target(f"""
SELECT printf('%d-%d%%', CAST(chance * 10 AS INTEGER) * 10, CAST(chance * 10 AS INTEGER) * 10 + 10) AS "Chance given",
       100.0 * AVG(hit) AS "Reached", 100.0 * AVG(chance) AS "Said", COUNT(*) AS "Predictions"
FROM summary_pick WHERE {scored}
GROUP BY CAST(chance * 10 AS INTEGER) ORDER BY CAST(chance * 10 AS INTEGER)""", time=False)],
        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "decimals": 0},
                        "overrides": [override("Predictions", unit="short", custom__hideFrom={
                            "viz": True, "legend": True, "tooltip": False}),
                                      override("Said", color={"mode": "fixed", "fixedColor": "text"}),
                                      override("Reached", color={"mode": "fixed", "fixedColor": "blue"})]},
        "options": {"xField": "Chance given", "orientation": "vertical", "showValue": "auto",
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
    }, 0, y, 12, 9)
    lay.add({
        "type": "barchart", "title": "Edge against what happened (live)", "datasource": SP,
        "description": "Scored predictions grouped by edge (chance minus the base rate, in points): how often they "
                       "reached the target, against the base rate for the same predictions. The held-out version "
                       "of this is below.",
        "targets": [target(f"""
SELECT CASE WHEN e < 0 THEN 'below 0' WHEN e < 5 THEN '0 to +5' WHEN e < 10 THEN '+5 to +10'
            WHEN e < 15 THEN '+10 to +15' WHEN e < 20 THEN '+15 to +20' ELSE '+20 or more' END AS "Edge",
       100.0 * AVG(hit) AS "Reached", 100.0 * AVG(base_rate) AS "Base rate", COUNT(*) AS "Predictions"
FROM (SELECT ROUND((chance - base_rate) * 100, 6) AS e, hit, base_rate FROM summary_pick
      WHERE {scored} AND base_rate IS NOT NULL)
GROUP BY 1 ORDER BY MIN(e)""", time=False)],
        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "decimals": 0},
                        "overrides": [override("Predictions", unit="short", custom__hideFrom={
                            "viz": True, "legend": True, "tooltip": False}),
                                      override("Base rate", color={"mode": "fixed", "fixedColor": "text"}),
                                      override("Reached", color={"mode": "fixed", "fixedColor": "green"})]},
        "options": {"xField": "Edge", "orientation": "vertical", "showValue": "auto",
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
    }, 12, y, 12, 9)
    y += 9
    lay.add({
        "type": "table", "title": "Track record per stock", "datasource": SP,
        "description": "Per stock, over the scored sessions in the time range: how often a run gave it 50%+ and it "
                       "then reached the target, the same for flagged, and how often the call (yes at 50%+, else "
                       "no) was right.",
        "targets": [target(f"""
SELECT symbol AS "Stock", COUNT(*) AS "Checks",
       SUM(passed = 1) AS "Said 50%+", SUM(CASE WHEN passed = 1 THEN hit ELSE 0 END) AS "…and reached",
       SUM(flagged = 1) AS "Flagged", SUM(CASE WHEN flagged = 1 THEN hit ELSE 0 END) AS "…reached ",
       100.0 * AVG(hit) AS "Reached at all %",
       100.0 * AVG(CASE WHEN (passed = 1) = (hit = 1) THEN 1.0 ELSE 0.0 END) AS "Right %"
FROM summary_pick WHERE {scored}
GROUP BY symbol ORDER BY SUM(flagged = 1) DESC, SUM(passed = 1) DESC, symbol""", time=False)],
        "fieldConfig": {"defaults": {"decimals": 0, "custom": {"filterable": True}}, "overrides": []},
        "options": {"showHeader": True, "cellHeight": "sm"},
    }, 0, y, 24, 10)
    y += 10

    # The latest check: the most recent month tested, then the latest recorded.
    latest_eval = (f"(SELECT model_version FROM model_evaluation WHERE family = '${{family}}' "
                   "ORDER BY sessions_to DESC, recorded_at DESC LIMIT 1)")
    lay.add({"type": "row", "title": "The held-out check: a twin of the model, tested on a month it had not seen",
             "collapsed": False, "panels": []}, 0, y, 24, 1)
    y += 1
    for i, (title, col, unit) in enumerate((("Any stock", "base_hit", "percentunit"),
                                             ("Model's top 5", "model_hit", "percentunit"),
                                             ("Most volatile 5", "volatile_hit", "percentunit"),
                                             ("Top 5, net a trade", "model_net", "percent"))):
        lay.add(stat(title, f"SELECT {col} AS \"{title}\" FROM model_evaluation WHERE family = '${{family}}' "
                     f"AND model_version = {latest_eval}", unit=unit, decimals=2 if unit == "percent" else 1,
                     desc="From the latest held-out check of this model: how often each set reached the target "
                          "(the net is after costs, selling at the target or the close)."), 6 * i, y, 6, 4)
    y += 4
    lay.add({
        "type": "barchart", "title": "Held-out record by edge", "datasource": SP,
        "description": "In the held-out month: how often stocks reached the target, by how far their chance sat "
                       "above the base rate at the same point. This is what today.sh's flag (+10 or more) quotes.",
        "targets": [target(f"""
SELECT CASE WHEN json_extract(b.value, '$.from') <= -100 THEN 'below 0'
            WHEN json_extract(b.value, '$.to') >= 100 THEN printf('%+d or more', json_extract(b.value, '$.from'))
            ELSE printf('%+d to %+d', json_extract(b.value, '$.from'), json_extract(b.value, '$.to')) END AS "Edge",
       100.0 * json_extract(b.value, '$.hit') AS "Reached", 100.0 * json_extract(b.value, '$.base') AS "Base rate",
       json_extract(b.value, '$.rows') AS "Checks"
FROM model_evaluation e, json_each(e.evaluation, '$.by_edge') b
WHERE e.model_version = {latest_eval}
ORDER BY json_extract(b.value, '$.from')""", time=False)],
        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "decimals": 0},
                        "overrides": [override("Checks", unit="short", custom__hideFrom={
                            "viz": True, "legend": True, "tooltip": False}),
                                      override("Base rate", color={"mode": "fixed", "fixedColor": "text"}),
                                      override("Reached", color={"mode": "fixed", "fixedColor": "green"})]},
        "options": {"xField": "Edge", "orientation": "vertical", "showValue": "auto",
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
    }, 0, y, 12, 9)
    lay.add({
        "type": "barchart", "title": "Held-out base rate and top 5, by time of day", "datasource": SP,
        "description": "In the held-out month, at each 15-minute check: how often any stock reached the target from "
                       "then, and the model's top 5 and the 5 most volatile. Empty for tomorrow.sh, which has one "
                       "point, the open.",
        "targets": [target(f"""
SELECT json_extract(p.value, '$.point') AS "Time",
       100.0 * json_extract(p.value, '$.base_hit') AS "Any stock",
       100.0 * json_extract(p.value, '$.model_hit') AS "Model's top 5",
       100.0 * json_extract(p.value, '$.intraday_volatility_hit') AS "Most volatile 5"
FROM model_evaluation e, json_each(e.evaluation, '$.by_point') p
WHERE e.model_version = {latest_eval}
ORDER BY json_extract(p.value, '$.point')""", time=False)],
        "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "decimals": 0},
                        "overrides": [override("Any stock", color={"mode": "fixed", "fixedColor": "text"}),
                                      override("Model's top 5", color={"mode": "fixed", "fixedColor": "orange"}),
                                      override("Most volatile 5", color={"mode": "fixed", "fixedColor": "purple"})]},
        "options": {"xField": "Time", "orientation": "vertical", "showValue": "never",
                    "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
    }, 12, y, 12, 9)

    return dashboard("stockpredictor-scorecard", "How the picks did",
                     "The recorded picks, scored after each session, and the held-out check behind the chances.",
                     lay.panels, templating, time_from="now-30d", time_to="now", refresh="5m")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, d in (("01-predicted-picks.json", predicted_picks()), ("02-how-the-picks-did.json", how_they_did())):
        (OUT / name).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {OUT / name}: {len(d['panels'])} panels")


if __name__ == "__main__":
    main()
