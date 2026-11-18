"""The Grafana dashboards: every query runs against the real schemas, and the
committed JSON is what grafana/make_dashboards.py writes."""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import subprocess
from datetime import date, time
from pathlib import Path

import pytest

from stock_predictor.clock import NSE
from stock_predictor.config import CONFIG
from stock_predictor.data.store import Store

ROOT = Path(__file__).resolve().parents[1]
DASHBOARDS = sorted((ROOT / "grafana" / "dashboards").glob("*.json"))
DAY = date(2026, 11, 13)
T0 = NSE.epoch_of(DAY, time(9, 20))
FROM_MS, TO_MS = NSE.epoch_of(DAY, time(0, 0)) * 1000, NSE.epoch_of(DAY, time(23, 59)) * 1000


def _queries(d: dict):
    """(where, datasource uid, sql) for every query in a dashboard."""
    for v in d["templating"]["list"]:
        yield f"variable {v['name']}", v["datasource"]["uid"], v["query"]
    for a in d["annotations"]["list"]:
        if "target" in a and "queryText" in a["target"]:
            yield f"annotation {a['name']}", a["datasource"]["uid"], a["target"]["queryText"]
    for p in d["panels"]:
        for t in p.get("targets", []):
            yield f"panel {p['title']!r}", t["datasource"]["uid"], t["queryText"]


HELD_OUT_STATS = {"panel 'Any stock'", "panel \"Model's top 5\"", "panel 'Most volatile 5'", "panel 'Top 5, net a trade'"}
EXPECTED_HELD_OUT_ROWS = {"panel 'Held-out record by edge'": 2, "panel 'Held-out base rate and top 5, by time of day'": 1}


def _fill(sql: str, family: str) -> str:
    sql = sql.replace("${family}", family).replace("${symbol:sqlstring}", "'DIVISLAB'")
    sql = sql.replace("${symbol}", "DIVISLAB").replace("$__from", str(FROM_MS)).replace("$__to", str(TO_MS))
    assert "$" not in re.sub(r"'\$\.[a-z_]+'", "", sql), sql  # only JSON paths like '$.by_edge' are left
    return sql


@pytest.fixture
def filled_store(tmp_path):
    s = Store(tmp_path / "sp.db")
    rows = []
    for i, (sym, chance) in enumerate((("DIVISLAB", 0.72), ("LT", 0.62), ("TMCV", 0.55))):
        for run in range(3):
            rows.append({"made_at": T0 + 600 * run, "kind": "today", "model_version": "sklearn-target-0.5pct-close/d",
                         "target_pct": 0.5, "session_date": DAY.isoformat(), "base_session": DAY.isoformat(),
                         "as_of": T0 + 600 * run, "symbol": sym, "rank": i + 1, "price": 100.0 + run,
                         "chance": chance, "p_model": chance, "gain_p25": 0.3, "gain_p50": 0.7, "gain_p75": 1.2,
                         "passed": int(chance >= 0.5), "stop_pct": None, "base_rate": 0.59,
                         "flagged": int(chance - 0.59 >= 0.1)})
    s.save_summary_picks(rows)
    s.score_summary_picks([{"id": r["id"], "entry_price": 100.0, "best_gain_pct": 0.8, "close_return_pct": 0.1,
                            "hit": int(r["symbol"] != "TMCV"), "scored_at": T0 + 10**5}
                           for r in s.summary_picks(DAY.isoformat())])
    # an older check recorded in the same second (as a backfill does): the dashboards must take the newer
    s.save_evaluation("sklearn-target-0.5pct-close/2026-10-08", "today", {
        "target": "+0.5% before the close", "sessions": ["2026-10-09", "2026-11-09"], "summary": {"base_hit": 0.5},
        "by_point": [{"point": "09:30", "base_hit": 0.7}], "by_edge": []}, recorded_at=T0)
    s.save_evaluation("sklearn-target-0.5pct-close/2026-10-14", "today", {
        "target": "+0.5% before the close", "sessions": ["2026-10-15", "2026-11-13"], "session_count": 21,
        "cost_pct": 0.083, "summary": {"base_hit": 0.516, "base_net": 0.01, "model": {"hit": 0.649, "net": -0.026},
                                       "intraday_volatility": {"hit": 0.662}},
        "by_point": [{"point": "09:30", "base_hit": 0.74, "model_hit": 0.9, "intraday_volatility_hit": 0.95}],
        "by_edge": [{"from": -100, "to": 0, "rows": 10, "hit": 0.5, "base": 0.5, "net": 0.0},
                    {"from": 10, "to": 15, "rows": 5, "hit": 0.6, "base": 0.4, "net": 0.02}]}, recorded_at=T0)
    yield s
    s.close()


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_every_predictor_query_runs(path, filled_store):
    d = json.loads(path.read_text())
    ran = 0
    for where, uid, sql in _queries(d):
        if uid != "stock-predictor-sqlite":
            continue
        rows = filled_store._conn.execute(_fill(sql, "sklearn-target-0.5pct-close")).fetchall()
        assert rows is not None, where
        if where.startswith("panel") and ("held-out" in where.lower() or where in HELD_OUT_STATS):
            assert len(rows) == EXPECTED_HELD_OUT_ROWS.get(where, 1), (where, rows)
        ran += 1
    assert ran >= 5


def test_the_per_stock_chart_has_the_range_the_entry_and_the_chance(filled_store):
    d = json.loads((ROOT / "grafana" / "dashboards" / "01-predicted-picks.json").read_text())
    panel = next(p for p in d["panels"] if p["type"] == "candlestick" and p["datasource"]["uid"] == "stock-predictor-sqlite")
    cur = filled_store._conn.execute(_fill(panel["targets"][0]["queryText"], "sklearn-target-0.5pct-close"))
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    assert cols[:8] == ["time", "open", "high", "low", "close", "Entry price", "Chance", "Price then"]
    assert len(rows) == 3 and rows[0]["time"] == T0
    first = rows[0]
    assert first["low"] == pytest.approx(100.3) and first["high"] == pytest.approx(101.2)  # the range in price
    assert first["Entry price"] == pytest.approx(min(100.0, 100.3 * 0.994))
    assert first["Chance"] == pytest.approx(72) and first["Edge"] == pytest.approx(13)
    assert panel["options"]["includeAllFields"] is True
    chance = next(o for o in panel["fieldConfig"]["overrides"] if o["matcher"]["options"] == "Chance")
    props = {p["id"]: p["value"] for p in chance["properties"]}
    assert props["custom.drawStyle"] == "bars" and props["custom.axisPlacement"] == "right"
    assert any(p.get("repeat") == "symbol" for p in d["panels"])  # one row of charts per stock picked


def test_clicking_a_stock_in_the_latest_run_sets_the_stocks_box():
    d = json.loads((ROOT / "grafana" / "dashboards" / "01-predicted-picks.json").read_text())
    assert d["title"] == "Predicted Picks" and d["uid"] == "stockpredictor-picks"  # renamed, same links
    table = next(p for p in d["panels"] if p["type"] == "table" and p["title"].startswith("Latest run"))
    stock = next(o for o in table["fieldConfig"]["overrides"] if o["matcher"]["options"] == "Stock")
    add, only = next(p["value"] for p in stock["properties"] if p["id"] == "links")
    for link in (add, only):
        assert link["url"].startswith("/d/stockpredictor-picks/predicted-picks?${__url_time_range}&${family:queryparam}")
        assert link["url"].endswith("&var-symbol=${__value.raw:percentencode}")  # M&M stays one stock
        assert link["targetBlank"] is False
    assert "&${symbol:queryparam}&" in add["url"]      # keeps the stocks already picked
    assert "${symbol:queryparam}" not in only["url"]   # replaces them
    names = {v["name"] for v in d["templating"]["list"]}
    assert {"family", "symbol"} <= names


def test_each_stocks_row_links_to_its_stock_analyzer_pages():
    d = json.loads((ROOT / "grafana" / "dashboards" / "01-predicted-picks.json").read_text())
    i = next(i for i, p in enumerate(d["panels"]) if p.get("repeat") == "symbol")
    bar = d["panels"][i + 1]  # first in the row, so it repeats with it, above the charts
    assert bar["type"] == "text" and bar["gridPos"]["y"] == d["panels"][i]["gridPos"]["y"] + 1
    assert all(p["gridPos"]["y"] > bar["gridPos"]["y"] for p in d["panels"][i + 2:])
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", bar["options"]["content"])
    assert [t for t, _ in links] == ["Company Details", "CAN SLIM", "Stock Trend"]
    for _, url in links:
        # this row's stock, percent-encoded so M&M stays one stock; their own time range, not this one's
        assert url.endswith("?var-symbol=${symbol:percentencode}") and "__url_time_range" not in url
    theirs = CONFIG.analyzer_db.parent.parent / "grafana" / "dashboards"
    if theirs.exists():  # each link names one of stock-analyzer's dashboards, which picks a single stock
        pages = {}
        for f in theirs.glob("*.json"):
            sa = json.loads(f.read_text())
            pages[f"/d/{sa['uid']}/{sa['title'].lower().replace(' ', '-')}"] = sa
        for title, url in links:
            sa = pages[url.split("?")[0]]
            assert sa["title"] == title
            symbol = next(v for v in sa["templating"]["list"] if v["name"] == "symbol")
            assert not symbol.get("multi")


@pytest.mark.skipif(not CONFIG.analyzer_db.exists(), reason="needs stock-analyzer's database")
def test_the_minute_candles_tooltip_has_ohlc_volume_emas_and_the_previous_close():
    from stock_predictor.data.analyzer import _connect_read_only

    d = json.loads((ROOT / "grafana" / "dashboards" / "01-predicted-picks.json").read_text())
    panel = next(p for p in d["panels"] if p["title"].endswith("what the price did (minute candles)"))
    assert panel["options"]["includeAllFields"] is True and panel["options"]["tooltip"]["mode"] == "multi"
    conn = _connect_read_only(CONFIG.analyzer_db)
    session, symbol = conn.execute("SELECT session_date, symbol FROM v_intraday_candles "
                                   "ORDER BY session_date DESC LIMIT 1").fetchone()
    day = date.fromisoformat(session)
    sql = panel["targets"][0]["queryText"].replace("${symbol}", symbol)
    sql = sql.replace("$__from", str(NSE.epoch_of(day, time(9, 15)) * 1000))
    sql = sql.replace("$__to", str(NSE.epoch_of(day, time(15, 30)) * 1000))
    cur = conn.execute(sql)
    cols = [c[0] for c in cur.description]
    assert cols == ["time", "Open", "Close", "High", "Low", "Volume", "EMA 10m", "EMA 60m", "Previous day's close"]
    rows = cur.fetchall()
    closes = [r[2] for r in rows]
    ema10, ema60 = closes[0], closes[0]
    for c in closes[1:]:
        ema10 += 2 / 11 * (c - ema10)
        ema60 += 2 / 61 * (c - ema60)
    assert rows[-1][6] == pytest.approx(ema10) and rows[-1][7] == pytest.approx(ema60)
    assert rows[8][6] is None and rows[9][6] is not None   # an average appears once its span has passed
    prev = conn.execute("SELECT close FROM v_daily_candles WHERE symbol = ? AND session_date < ? "
                        "ORDER BY session_date DESC LIMIT 1", (symbol, session)).fetchone()
    assert prev is None or {r[8] for r in rows} == {prev[0]}
    conn.close()


def test_prices_show_full_rupees_and_paise():
    """As Candles and Calls: the locale unit with 2 decimals (11,053.15), never a currency unit,
    which Grafana abbreviates above 999 (₹11.1K)."""
    text = (ROOT / "grafana" / "dashboards" / "01-predicted-picks.json").read_text()
    assert "currency" not in text
    d = json.loads(text)
    for p in d["panels"]:
        if p["type"] == "candlestick":
            assert (p["fieldConfig"]["defaults"]["unit"], p["fieldConfig"]["defaults"]["decimals"]) == ("locale", 2)
    table = next(p for p in d["panels"] if p["type"] == "table")
    units = {o["matcher"]["options"]: {x["id"]: x["value"] for x in o["properties"]}
             for o in table["fieldConfig"]["overrides"]}
    for col in ("Price", "Range low", "Range high", "Entry"):
        assert units[col] == {"unit": "locale", "decimals": 2}, col


def test_predicted_picks_opens_on_todays_trading_hours_with_stock_analyzers_time_options():
    d = json.loads((ROOT / "grafana" / "dashboards" / "01-predicted-picks.json").read_text())
    ranges = d["timepicker"]["quick_ranges"]
    assert ranges[0] == {"display": "Today's trading hours", "from": "now/d+9h+15m", "to": "now"}
    assert d["time"] == {"from": "now/d+9h+15m", "to": "now"}  # selected by default
    assert d["weekStart"] == "monday" and d["timezone"] == "Asia/Kolkata"
    theirs = CONFIG.analyzer_db.parent.parent / "grafana" / "dashboards" / "02-candles-and-calls.json"
    if theirs.exists():  # the same options, in the same order, as stock-analyzer's Candles and Calls
        assert ranges == json.loads(theirs.read_text())["timepicker"]["quick_ranges"]


@pytest.mark.skipif(not CONFIG.analyzer_db.exists(), reason="needs stock-analyzer's database")
@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_every_analyzer_query_runs_read_only(path):
    from stock_predictor.data.analyzer import _connect_read_only

    conn = _connect_read_only(CONFIG.analyzer_db)
    for where, uid, sql in _queries(json.loads(path.read_text())):
        if uid == "stock-analyzer-readonly":
            conn.execute(_fill(sql, "sklearn-target-0.5pct-close")).fetchall()
    conn.close()


def test_dashboards_are_consistent_and_up_to_date(tmp_path, monkeypatch):
    uids = {"stock-predictor-sqlite", "stock-analyzer-readonly", "-- Grafana --"}
    for path in DASHBOARDS:
        d = json.loads(path.read_text())
        ids = [p["id"] for p in d["panels"]]
        assert len(ids) == len(set(ids)), path.name
        assert {u for _, u, _ in _queries(d)} <= uids
    provisioned = (ROOT / "grafana" / "provisioning" / "datasources" / "stock-predictor.yaml").read_text()
    assert "uid: stock-predictor-sqlite" in provisioned and "uid: stock-analyzer-readonly" in provisioned
    assert provisioned.count("query_only(1)") == 2  # neither database can be changed from a dashboard
    # added to stock-analyzer's Grafana, whose own datasource is the default: Grafana allows only one
    assert "isDefault: true" not in provisioned and provisioned.count("isDefault: false") == 2
    folder = (ROOT / "grafana" / "provisioning" / "dashboards" / "stock-predictor.yaml").read_text()
    assert "folder: Stock Predictor" in folder and "path: __DASHBOARDS__" in folder

    spec = importlib.util.spec_from_file_location("make_dashboards", ROOT / "grafana" / "make_dashboards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    mod.main()
    for path in DASHBOARDS:
        assert (tmp_path / path.name).read_text() == path.read_text(), f"re-run grafana/make_dashboards.py ({path.name})"


@pytest.mark.parametrize("script", ["schedule-today.sh", "../grafana/add-to-stock-analyzer.sh"])
def test_the_new_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)]).returncode == 0


def test_the_schedule_runs_every_ten_minutes_on_weekdays():
    out = subprocess.run(["bash", str(ROOT / "scripts" / "schedule-today.sh"), "plist", "--stop", "0.5"],
                         capture_output=True, text=True, check=True).stdout
    slots = re.findall(r"<key>Weekday</key><integer>(\d)</integer><key>Hour</key><integer>(\d+)</integer>"
                       r"<key>Minute</key><integer>(\d+)</integer>", out)
    assert len(slots) == 5 * 34
    times = sorted({(int(h), int(m)) for _, h, m in slots})
    assert times[0] == (9, 20) and times[-1] == (14, 50) and len(times) == 34
    assert {d for d, _, _ in slots} == {"1", "2", "3", "4", "5"}
    assert "<string>run</string>\n    <string>--stop</string>\n    <string>0.5</string>" in out


def test_held_out_checks_are_kept_in_the_store(tmp_path):
    with Store(tmp_path / "sp.db") as s:
        s.save_evaluation("fam/2026-10-14", "today", {"summary": {"base_hit": 0.5, "model": {"hit": 0.6}},
                                                      "sessions": ["a", "b"], "by_edge": []})
        s.save_evaluation("fam/2026-10-14", "today", {"summary": {"base_hit": 0.4}, "sessions": ["a", "b"]})
        (e,) = s.evaluations()
        assert e["family"] == "fam" and e["base_hit"] == 0.4  # replaced, not duplicated
        assert json.loads(e["evaluation"])["sessions"] == ["a", "b"]


def test_adding_to_stock_analyzers_grafana_writes_two_files_and_removes_them(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "provisioning" / "datasources").mkdir(parents=True)
    (runtime / "provisioning" / "dashboards").mkdir(parents=True)
    (runtime / "provisioning" / "datasources" / "sqlite.yaml").write_text("theirs")  # never touched
    dbs = tmp_path / "sp.db", tmp_path / "sa.db"
    for db in dbs:
        db.write_text("")
    env = {"PATH": "/usr/bin:/bin", "STOCK_ANALYZER_GRAFANA_RUNTIME": str(runtime),
           "STOCK_PREDICTOR_DB": str(dbs[0]), "STOCK_ANALYZER_DB": str(dbs[1])}
    script = str(ROOT / "grafana" / "add-to-stock-analyzer.sh")
    out = subprocess.run(["bash", script], env=env, capture_output=True, text=True, check=True).stdout
    assert "Restart stock-analyzer's Grafana" in out  # no password given: nothing is reloaded
    ds = (runtime / "provisioning" / "datasources" / "stock-predictor.yaml").read_text()
    folder = (runtime / "provisioning" / "dashboards" / "stock-predictor.yaml").read_text()
    assert f"path: {dbs[0]}" in ds and f"path: {dbs[1]}" in ds and "__" not in ds
    assert f"path: {ROOT / 'grafana' / 'dashboards'}" in folder
    assert (runtime / "provisioning" / "datasources" / "sqlite.yaml").read_text() == "theirs"
    subprocess.run(["bash", script, "remove"], env=env, capture_output=True, text=True, check=True)
    assert sorted(p.name for p in (runtime / "provisioning").rglob("*.yaml")) == ["sqlite.yaml"]
