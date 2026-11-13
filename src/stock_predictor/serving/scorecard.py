"""How recorded picks did: what today.sh and tomorrow.sh said about a session,
against what the session then did.

Every summary run (``summary --record``, which both scripts pass) stores every
stock's chance, expected gain and reference price. A pick is scored against
the session's own candles:

* **today** (``today.sh``): entry is the price at the minute the prediction
  used; it hit if a later high in the session reached entry × (1 + target).
* **next** (``tomorrow.sh``): entry is the session's open; it hit if the
  session's high reached open × (1 + target).

Outcomes are saved once the session has closed. Before that, the scorecard
reports how things stand so far: a pick that has reached its target already
counts, one that has not is still open.
"""

from __future__ import annotations

import time
from datetime import date
from statistics import median
from typing import Any, Optional

from stock_predictor.clock import NSE


def _candles_by_symbol(views, session: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for c in views.intraday_candles(session):
        out.setdefault(c.symbol, []).append(c)
    return out


def score_pick(pick: dict, candles: list) -> Optional[dict]:
    """The outcome of one recorded pick, or None with no candles to judge by."""
    if pick["kind"] == "next":
        after = candles
        entry = candles[0].open if candles else None
    else:
        after = [c for c in candles if c.ts_epoch > pick["as_of"]]
        entry = pick["price"]
    if not after or not entry or entry <= 0:
        return None
    best = max(max(c.high for c in after) / entry - 1.0, 0.0) * 100.0
    return {"id": pick.get("id"), "entry_price": float(entry), "best_gain_pct": best,
            "close_return_pct": (after[-1].close / entry - 1.0) * 100.0,
            "hit": int(best >= pick["target_pct"] - 1e-9)}


def scorecard(views, store, session: str, now: Optional[int] = None, persist: bool = True) -> dict[str, Any]:
    """Every recorded run about ``session``, scored."""
    now = now or int(time.time())
    picks = store.summary_picks(session)
    closed = now >= NSE.session_close_epoch(date.fromisoformat(session)) + 60 or session in views.sessions()
    report: dict[str, Any] = {"session": session, "closed": closed, "runs": []}
    if not picks:
        return report
    candles = _candles_by_symbol(views, session)
    runs: dict[tuple, list[dict]] = {}
    for p in picks:
        runs.setdefault((p["kind"], p["model_version"], p["as_of"]), []).append(p)
    to_save = []
    for (kind, version, as_of), rows in sorted(runs.items(), key=lambda kv: (kv[0][0] != "next", kv[0][2])):
        scored = []
        for p in rows:
            o = score_pick(p, candles.get(p["symbol"], []))
            if o is None:
                continue
            scored.append({**p, **o})
            if closed and persist and p["scored_at"] is None:
                to_save.append({**o, "id": p["id"], "scored_at": now})
        if not scored:
            continue
        t = rows[0]["target_pct"]
        passed = [r for r in scored if r["passed"]]
        top5 = sorted(scored, key=lambda r: r["rank"])[:5]

        def rate(rs):
            return sum(r["hit"] for r in rs) / len(rs) if rs else None

        report["runs"].append({
            "kind": kind, "model_version": version, "as_of": as_of,
            "as_of_time": NSE.time_of(as_of).strftime("%H:%M"),
            "base_session": rows[0]["base_session"], "target_pct": t,
            "made_at": min(r["made_at"] for r in rows),
            "stocks": len(scored), "every_stock_hit": rate(scored),
            "passed": len(passed), "passed_hits": sum(r["hit"] for r in passed), "passed_hit": rate(passed),
            "passed_chance": (sum(r["chance"] for r in passed if r["chance"] is not None) / len(passed)) if passed else None,
            "predicted_median": median(r["gain_p50"] for r in passed) if passed else None,
            "actual_median": median(r["best_gain_pct"] for r in passed) if passed else None,
            "top5_hits": sum(r["hit"] for r in top5),
            "top5": [{"symbol": r["symbol"], "chance": r["chance"], "predicted_median": r["gain_p50"],
                      "best_gain_pct": r["best_gain_pct"], "close_return_pct": r["close_return_pct"],
                      "hit": bool(r["hit"])} for r in top5],
        })
    if to_save:
        store.score_summary_picks(to_save)
    report["saved"] = len(to_save)
    return report


def track_records(store, since: str, until: Optional[str] = None) -> dict[str, dict]:
    """Per stock, from scored recorded picks with ``since <= session <= until``:
    how often a run said 50%+, how often it then reached the target, and how
    often the call (yes or no) was right. Kept apart by kind."""
    sql = ("SELECT kind, symbol, passed, hit FROM summary_pick WHERE scored_at IS NOT NULL "
           "AND session_date >= ?")
    args: list = [since]
    if until:
        sql += " AND session_date <= ?"
        args.append(until)
    out: dict[str, dict] = {}
    for kind, sym, passed, hit in store._conn.execute(sql, args):
        r = out.setdefault(kind, {}).setdefault(sym, {"checks": 0, "said_yes": 0, "yes_reached": 0,
                                                       "right": 0, "reached": 0})
        r["checks"] += 1
        r["said_yes"] += passed
        r["yes_reached"] += passed and hit
        r["right"] += int(bool(passed) == bool(hit))
        r["reached"] += hit
    return out


def format_track_records(records: dict[str, dict], since: str) -> str:
    if not records:
        return (f"No scored picks since {since} yet: today.sh and tomorrow.sh record every run, and "
                "the scorecard scores them once the session has closed.")
    names = {"today": "today.sh (from the price at each run)", "next": "tomorrow.sh (from the next open)"}
    lines = [f"Track record of recorded picks since {since}"]
    for kind, per in records.items():
        lines += ["", f"  {names.get(kind, kind)}:  stock, said 50%+ / reached, right (yes or no), checks"]
        for sym, r in sorted(per.items(), key=lambda kv: (-kv[1]["said_yes"], kv[0])):
            right = r["right"] / r["checks"] if r["checks"] else 0
            lines.append(f"    {sym:<12} {r['yes_reached']:>3}/{r['said_yes']:<3}  {right:>4.0%}  {r['checks']:>4}")
    return "\n".join(lines)


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def format_scorecard(r: dict[str, Any]) -> str:
    s = r["session"]
    if not r["runs"]:
        return (f"How {s}'s picks did: nothing recorded for {s} "
                "(today.sh and tomorrow.sh record every run).")
    state = "final" if r["closed"] else "so far: the session is still running, so a miss may yet turn"
    lines = [f"How {s}'s picks did ({state})"]
    for run in r["runs"]:
        t = run["target_pct"]
        if run["kind"] == "next":
            head = (f"  tomorrow.sh on {run['base_session']} (state at {run['as_of_time']}): "
                    f"+{t:g}% above {s}'s open")
        else:
            head = f"  today.sh at {run['as_of_time']}: +{t:g}% above the price then"
        lines.append(head + f" · {run['model_version']}")
        if run["passed"]:
            lines.append(
                f"    {run['passed']} picks with a chance of 50%+ (said {_pct(run['passed_chance'])} on average): "
                f"{run['passed_hits']} reached it ({_pct(run['passed_hit'])}); every stock: "
                f"{_pct(run['every_stock_hit'])}.")
            lines.append(
                f"    Median expected gain {run['predicted_median']:+.2f}%; actual median best gain "
                f"{run['actual_median']:+.2f}%.")
        else:
            lines.append(f"    No stock had a chance of 50%+; every stock: {_pct(run['every_stock_hit'])}.")
        top = ", ".join(f"{p['symbol']} {'✓' if p['hit'] else '✗'} {p['best_gain_pct']:+.2f}%" for p in run["top5"])
        lines.append(f"    Top 5: {run['top5_hits']} of 5 · {top}")
    return "\n".join(lines)
