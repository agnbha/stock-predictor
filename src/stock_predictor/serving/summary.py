"""The on-demand summary: which stocks are expected to gain at least the
target before today's close, by how much, and how far to trust that.

``stock-predictor summary`` prints it; ``scripts/today.sh`` runs that after
making sure a current model exists. Everything comes from one model trained
with ``--target PCT --to-close --band max-gain``:

* **chance**: P(the price trades PCT% above the price now before the close),
  calibrated on recent sessions the calibration fit had not seen;
* **expected gain**: the best price later today against the price now, as a
  median and a p25-p75 range. It is the most a seller could have got, not
  where the stock closes, and it is before costs.

"How much to trust it" comes from the held-out check kept beside a model
trained on the same target without the last month (``evaluate-target
--holdout --record``).
"""

from __future__ import annotations

import math
import textwrap
import time
from dataclasses import asdict
from datetime import date
from typing import Any, Optional

import numpy as np

from stock_predictor.clock import NSE
from stock_predictor.config import LAST_INTRADAY_POINT
from stock_predictor.models.registry import ModelRegistry, card_target
from stock_predictor.models.triple_barrier import TripleBarrierConfig


class SummaryUnavailable(RuntimeError):
    """Nothing to summarise yet: no model for the target, or no candles."""


def target_config(target_pct: float) -> TripleBarrierConfig:
    return TripleBarrierConfig(target_pct=target_pct, to_close=True)


def find_model(registry: ModelRegistry, target_pct: float, recent: bool = False,
               next_session: bool = False) -> Optional[str]:
    """The to-the-close model for ``target_pct`` trained through the latest
    session (of those, the one saved last): trained on every session, or with
    ``recent`` on only the last few weeks (``train --recent-days``). With
    ``next_session``, the model for the session after (``train --next-session``)."""
    want = asdict(target_config(target_pct))
    entry = "next_open" if next_session else "minute"
    found = [c for c in registry.cards()
             if c.target == want and c.band == "max_gain_pct" and bool(c.recent_days) == recent
             and c.entry == entry]
    return max(found, key=lambda c: c.trained_through).model_version if found else None


def model_state(target_pct: float, registry: Optional[ModelRegistry] = None, views=None,
                recent: bool = False, next_session: bool = False) -> str:
    """'missing', 'stale' (trained before the latest finished session) or 'ok'."""
    from stock_predictor.data.analyzer import AnalyzerViews

    registry = registry or ModelRegistry()
    version = find_model(registry, target_pct, recent, next_session)
    if version is None:
        return "missing"
    own = views is None
    views = views or AnalyzerViews()
    try:
        finished = views.sessions()
    finally:
        if own:
            views.close()
    card = registry.card(version)
    return "stale" if finished and card.trained_through < finished[-1] else "ok"


def score_symbols(model, views, session: str, as_of: int, params=None):
    """Every symbol's latest row at ``as_of``, scored: price, chance (model and
    calibrated), the band (p25/p50/p75) and its volatility rank."""
    import pandas as pd

    from stock_predictor.features.pipeline import build_feature_frame, latest_row_per_symbol

    frame = build_feature_frame(views, session, as_of, params=params)
    if frame.empty:
        return pd.DataFrame()
    rows = latest_row_per_symbol(frame)
    pred = model.predict(rows)
    out = pd.DataFrame({
        "symbol": rows["symbol"].to_numpy(), "last_ts": rows["ts_epoch"].to_numpy(),
        "price": rows["_close"].to_numpy(dtype=float), "p_model": pred.p_up,
        "p_seen": pred.p_seen if pred.p_seen is not None else np.nan,
        "p25": pred.gain_p25, "p50": pred.gain_p50, "p75": pred.gain_p75,
        "vol_pct": (rows["atr_14"] / rows["_close"].replace(0, np.nan) * 100).to_numpy(),
    })
    out["chance"] = out["p_seen"].fillna(out["p_model"])
    out["vol_rank"] = out["vol_pct"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["chance", "p50"], ascending=False).reset_index(drop=True)


def build_summary(rows, cfg: TripleBarrierConfig, version: str, card, session: str, as_of: int,
                  finished: Optional[int], latest_session: Optional[str], evaluation: Optional[dict],
                  min_chance: float = 0.5, top: int = 15, now: Optional[int] = None,
                  capped: bool = False, live: bool = True,
                  next_session: Optional[dict] = None) -> dict[str, Any]:
    """Everything the summary says, as plain data (``--json`` writes it)."""
    now = now or int(time.time())
    target = cfg.target_pct
    passed = rows[rows["chance"] >= min_chance].reset_index(drop=True)
    head, rest = passed.head(top), passed.iloc[top:]

    record = (evaluation or {}).get("by_symbol") or {}

    def row(r) -> dict:
        return {"symbol": r["symbol"], "price": float(r["price"]), "chance": float(r["chance"]),
                "track": record.get(r["symbol"]),
                "p_model": float(r["p_model"]), "median_gain_pct": float(r["p50"]),
                "range_pct": [float(r["p25"]), float(r["p75"])],
                # the same range as prices: the price shown, raised by the p25 and p75 gains
                "range_price": [float(r["price"] * (1 + r["p25"] / 100)), float(r["price"] * (1 + r["p75"] / 100))],
                "target_price": float(r["price"] * (1 + target / 100)), "vol_rank": int(r["vol_rank"])}

    groups = []
    if len(rest):
        floor = rest["p50"].map(lambda x: math.floor(round(x * 10, 6)) / 10)
        for lo in sorted(floor.unique(), reverse=True):
            g = rest[floor == lo].sort_values("p50", ascending=False)
            groups.append({"from_pct": float(lo), "to_pct": float(lo + 0.09),
                           "symbols": list(g["symbol"])})

    s: dict[str, Any] = {
        "target_pct": target, "question": cfg.describe(), "session": session, "live": live,
        "next_session": next_session,
        "as_of": as_of, "as_of_time": NSE.time_of(as_of).strftime("%H:%M"),
        "lag_minutes": max(0, (now - finished - 60) // 60) if finished and live else None,
        "model_version": version, "trained_through": card.trained_through,
        "trained_from": getattr(card, "trained_from", "") or None,
        "recent_days": getattr(card, "recent_days", None),
        "minute_step": getattr(card, "minute_step", 1),
        "train_sessions": card.train_sessions, "calibrated": bool(rows["p_seen"].notna().any()),
        "min_chance": min_chance, "universe": int(len(rows)), "passed": int(len(passed)),
        "top": [row(r) for _, r in head.iterrows()],
        # every stock, in rank order, for recording (summary --record) and --json
        "all": [{**row(r), "rank": i + 1, "passed": bool(r["chance"] >= min_chance),
                 "p25": float(r["p25"]), "p50": float(r["p50"]), "p75": float(r["p75"])}
                for i, (_, r) in enumerate(rows.iterrows())],
        "track_record": {"sessions": (evaluation or {}).get("sessions"),
                         "model_version": (evaluation or {}).get("model_version")} if record else None,
        "rest": {"count": int(len(rest)),
                 "chance_range": [float(rest["chance"].min()), float(rest["chance"].max())] if len(rest) else None,
                 "groups": groups,
                 "rows": [row(r) for _, r in rest.iterrows()]},
        "standouts": {
            "whole_range_above_target": list(passed.loc[passed["p25"] >= target, "symbol"]),
            "median_twice_target": list(passed.loc[passed["p50"] >= 2 * target, "symbol"]),
        },
        "trust": None,
        "notes": [],
    }
    if evaluation:
        hhmm = s["as_of_time"]
        at = [p for p in evaluation.get("by_point", []) if p["point"] <= hhmm]
        sm = evaluation["summary"]
        filt = {f["rule"]: f for f in evaluation.get("filters", [])}
        s["trust"] = {
            "model_version": evaluation.get("model_version"),
            "sessions": evaluation.get("sessions"), "session_count": evaluation.get("session_count"),
            "base_hit": sm["base_hit"], "base_hit_now": at[-1]["base_hit"] if at else None,
            "point": at[-1]["point"] if at else None,
            "model_top5_hit": sm["model"]["hit"], "volatile_top5_hit": sm["intraday_volatility"]["hit"],
            "volatile_label": sm.get("labels", {}).get("intraday_volatility", "the 5 most volatile stocks"),
            "volatile_top5_net": sm["intraday_volatility"]["net"],
            "model_top5_net": sm["model"]["net"], "base_net": sm["base_net"],
            "cost_pct": evaluation.get("cost_pct"),
            "beat_median": 1 - evaluation["band_coverage"]["p50"] if evaluation.get("band_coverage") else None,
            "calibrated_filter": filt.get("chance >= 50% (calibrated)"),
        }
    else:
        s["notes"].append("No held-out check for this target yet: run scripts/today.sh --evaluate.")
    judged = next_session["date"] if next_session else session
    if judged <= card.trained_through:
        s["notes"].append(f"The model was trained on sessions through {card.trained_through}, including "
                          f"{judged}: these figures are in-sample and flatter it.")
    if next_session and next_session.get("partial"):
        s["notes"].append(f"{session} has only reached {s['as_of_time']}; the model learnt from each stock's "
                          "state at 15:00. Run it again after 15:00 for the picture it was trained on.")
    if next_session and not next_session.get("known", True):
        s["notes"].append("No NSE holiday list for that year: the next session is taken to be the next weekday.")
    if latest_session and card.trained_through < latest_session:
        s["notes"].append(f"The model was trained through {card.trained_through}; {latest_session} has "
                          "finished since. Run scripts/today.sh --retrain for a current one.")
    if capped and not next_session:
        s["notes"].append(f"After {LAST_INTRADAY_POINT:%H:%M} the model has nothing to go on (it was "
                          f"trained to {LAST_INTRADAY_POINT:%H:%M}), so this is the picture at {s['as_of_time']}.")
    if s["lag_minutes"] is not None and s["lag_minutes"] >= 5 and not capped:
        s["notes"].append(f"stock-analyzer's candles are {s['lag_minutes']} minutes behind the clock.")
    if not s["calibrated"]:
        s["notes"].append("This model has no calibration: chance is its raw probability.")
    return s


def _pct(x: Optional[float], d: int = 0) -> str:
    return "-" if x is None else f"{x * 100:.{d}f}%"


def format_summary(s: dict[str, Any], width: int = 100) -> str:
    if s.get("next_session"):
        return _format_next(s, width)
    t = s["target_pct"]
    lines = [
        f"Expected to gain +{t:g}% or more before {'today' if s.get('live', True) else s['session']}'s close",
        f"{s['session']} · prices at {s['as_of_time']} IST"
        + (f" ({s['lag_minutes']} min behind the clock)" if s["lag_minutes"] else "")
        + f" · {s['model_version']}",
        "Trained on " + (f"the last {s['recent_days']} days: " if s.get("recent_days") else "")
        + f"{s['train_sessions']} sessions"
        + (f", {s['trained_from']} to {s['trained_through']}" if s.get("trained_from") else f" through {s['trained_through']}")
        + (", every minute" if s.get("minute_step", 1) == 1 else f", every {s['minute_step']} minutes") + ".",
        "",
        f"{s['passed']} of {s['universe']} stocks have a {_pct(s['min_chance'])} or better chance of trading "
        f"{t:g}% above the price now before 15:29.",
        "Expected gain: the best price later today against now (median, and the p25-p75 range), "
        "before costs. Where it closes can be lower.",
        "Range in price: the same p25-p75 range as prices, the best price expected later today.",
        "",
    ]
    lines += _table(s, "Price")
    lines += _rest_and_trust(s, width)
    return "\n".join(lines)


def _track(t: Optional[dict]) -> str:
    if not t:
        return ""
    return f"{t['yes_reached']:>3}/{t['said_yes']:<3} {t['right'] / t['checks'] * 100:>3.0f}%" if t["checks"] else ""


def _table(s: dict[str, Any], price_label: str) -> list[str]:
    if not s["top"]:
        return ["  (none)"]
    tr = s.get("track_record")
    money = [f"{a:.2f} to {b:.2f}" for a, b in (r["range_price"] for r in s["top"])]
    mw = max(len("Range in price"), *map(len, money))
    head = (f"  {'#':>2}  {'Stock':<12} {price_label:>10}  {'Chance':>6}  {'Median':>7}   {'Range':<17}"
            f"  {'Range in price':<{mw}}")
    if tr:
        head += "  Last month"
    out = [head]
    for i, r in enumerate(s["top"], start=1):
        lo, hi = r["range_pct"]
        line = (f"  {i:>2}  {r['symbol']:<12} {r['price']:>10.2f}  {_pct(r['chance']):>6}  "
                f"{r['median_gain_pct']:>+6.2f}%   {f'{lo:+.2f} to {hi:+.2f}%':<17}  {money[i - 1]:<{mw}}")
        if tr:
            line += "  " + _track(r.get("track"))
        out.append(line)
    if tr:
        sess = tr["sessions"] or ["?", "?"]
        per = "session" if s.get("next_session") else "15-minute check"
        out += ["", f"  Last month ({sess[0]} to {sess[1]}, a model that had not seen it): \"14/18\" means it said "
                    f"50%+ at 18 {per}s and the stock reached the target at 14; the % is how often its call, "
                    "yes or no, was right."]
    return out


def _format_next(s: dict[str, Any], width: int) -> str:
    t, nd = s["target_pct"], s["next_session"]["date"]
    lines = [
        f"Expected to gain +{t:g}% or more on {nd} (the next session), from its open",
        f"From each stock's state at {s['as_of_time']} IST on {s['session']}"
        + (f" ({s['lag_minutes']} min behind the clock)" if s["lag_minutes"] else "")
        + f" · {s['model_version']}",
        f"Trained on {s['train_sessions']} sessions"
        + (f", {s['trained_from']} to {s['trained_through']}" if s.get("trained_from") else "")
        + ": each stock's 15:00 state against the session after.",
        "",
        f"{s['passed']} of {s['universe']} stocks have a {_pct(s['min_chance'])} or better chance of trading "
        f"{t:g}% above {nd}'s open at some point before its close.",
        "Expected gain: the best price that session against its open (median, and the p25-p75 range), "
        "before costs. An overnight gap is not counted; the price column is the latest one.",
        "Range in price: the same p25-p75 range as prices, worked from the latest price "
        "(the open is not known yet, so it moves with wherever the stock opens).",
        "",
    ]
    lines += _table(s, "Price now")
    lines += _rest_and_trust(s, width)
    wide = s["next_session"].get("widest_range") or []
    if wide:
        at = lines.index("Notes:") - 1 if "Notes:" in lines else len(lines)
        lines[at:at] = ["", f"The {len(wide)} with the widest range on {s['session']} so far (the screen that did "
                            "best in testing): " + ", ".join(f"{w['symbol']} {w['range_pct']:.1f}%" for w in wide) + "."]
    return "\n".join(lines)


def _rest_and_trust(s: dict[str, Any], width: int) -> list[str]:
    t = s["target_pct"]
    nxt = bool(s.get("next_session"))
    lines: list[str] = []
    rest = s["rest"]
    if rest["count"]:
        lo, hi = rest["chance_range"]
        lines += ["", f"The other {rest['count']} (chance {_pct(lo)}-{_pct(hi)}), by median expected gain:"]
        for g in rest["groups"]:
            label = f"  {g['from_pct']:+.2f} to {g['to_pct']:+.2f}%  "
            body = textwrap.wrap(", ".join(g["symbols"]), width=width - len(label))
            lines.append(label + body[0])
            lines += [" " * len(label) + b for b in body[1:]]
    so = s["standouts"]
    stand = []
    if so["whole_range_above_target"]:
        stand.append(f"whole range at or above +{t:g}%: {', '.join(so['whole_range_above_target'])}")
    if so["median_twice_target"]:
        stand.append(f"median +{2 * t:g}% or more: {', '.join(so['median_twice_target'])}")
    if stand:
        lines += ["", "Stand-outs: " + "; ".join(stand) + "."]

    tr = s["trust"]
    if tr:
        sess = tr["sessions"] or ["?", "?"]
        lines += ["", f"How much to trust it (held out: {tr['session_count']} sessions, {sess[0]} to {sess[1]}, "
                      "with a model trained without them):"]
        now = (f" ({_pct(tr['base_hit_now'])} from {tr['point']})"
               if tr["base_hit_now"] is not None and not nxt else "")
        lines.append(f"  Any stock reached +{t:g}%{' from the open' if nxt else ''} "
                     f"{_pct(tr['base_hit'])} of the time{now}.")
        lines.append(f"  The model's top 5: {_pct(tr['model_top5_hit'])}; {tr.get('volatile_label', 'the 5 most volatile stocks')}: "
                     f"{_pct(tr['volatile_top5_hit'])} ({tr.get('volatile_top5_net', 0):+.2f}% a trade).")
        lines.append(f"  Buying the top 5{' at the open' if nxt else ''}, selling at +{t:g}% or else at the close: "
                     f"{tr['model_top5_net']:+.2f}% a trade after {tr['cost_pct']:.3f}% costs; "
                     f"buying every stock: {tr['base_net']:+.2f}%.")
        if tr["calibrated_filter"]:
            f = tr["calibrated_filter"]
            per = "a session" if nxt else "a check"
            lines.append(f"  There, {f['per_session_point']:.1f} stocks {per} had a calibrated chance of 50%+, and "
                         f"{_pct(f['hit'])} of them made it; now {s['passed']} do.")
        if tr["beat_median"] is not None:
            lines.append(f"  The actual best gain beat the median {_pct(tr['beat_median'])} of the time "
                         "(50% if the range were right).")
    if s["notes"]:
        lines += ["", "Notes:"] + [f"  ! {n}" for n in s["notes"]]
    return lines


def summarise(target_pct: float = 0.5, min_chance: float = 0.5, top: int = 15,
              model_version: Optional[str] = None, session: Optional[str] = None,
              at: Optional[int] = None, registry: Optional[ModelRegistry] = None, views=None,
              recent: bool = False, next_session: bool = False) -> dict:
    """Load the model, score the session in progress (or ``session``), and
    build the summary. With ``next_session``, score the session after it from
    each stock's state at 15:00 (or the latest finished session's, before the
    open)."""
    from stock_predictor.data.analyzer import AnalyzerViews
    from stock_predictor.params import Params

    registry = registry or ModelRegistry()
    version = model_version or find_model(registry, target_pct, recent, next_session)
    if version is None:
        which = ("the last-month " if recent else "") + ("next-session " if next_session else "")
        raise SummaryUnavailable(
            f"no {which}model for +{target_pct:g}%: run scripts/{'tomorrow' if next_session else 'today'}.sh "
            f"{'--recent ' if recent else ''}--retrain")
    model, card = registry.load(version)
    cfg = card_target(card)
    params = Params.from_dict(card.params) if card.params else None
    found = registry.extra_like(card, "evaluation")
    own = views is None
    views = views or AnalyzerViews(live=True)
    if next_session:
        try:
            return _summarise_next(model, card, cfg, version, params, found, views, session, at,
                                   min_chance, top)
        finally:
            if own:
                views.close()
    try:
        session = session or views.session_in_progress() or NSE.session_date_of(int(time.time())).isoformat()
        finished = views.last_finished_minute(session)
        if finished is None:
            raise SummaryUnavailable(f"no candles for {session} yet (the market opens at 09:15 IST)")
        last_point = NSE.epoch_of(date.fromisoformat(session), LAST_INTRADAY_POINT)
        as_of = at if at is not None else min(finished, last_point)
        rows = score_symbols(model, views, session, as_of, params)
        if rows.empty:
            raise SummaryUnavailable(f"no features for {session} at {NSE.time_of(as_of):%H:%M}")
        done = [s for s in views.sessions() if s < session]
        live = session == views.session_in_progress()
    finally:
        if own:
            views.close()
    return build_summary(rows, cfg, version, card, session, as_of, finished, done[-1] if done else None,
                         found[1] if found else None, min_chance, top,
                         capped=at is None and finished > last_point, live=live)


def _summarise_next(model, card, cfg, version, params, found, views, session, at, min_chance, top) -> dict:
    """The session after ``session`` (default: today if it has candles, else
    the latest finished session), from each stock's state at 15:00."""
    from stock_predictor.models.nextday import next_trading_day

    today = views.session_in_progress()
    if session is None:
        session = today if today and views.last_finished_minute(today) is not None else None
    finished_sessions = views.sessions()
    if session is None:
        if not finished_sessions:
            raise SummaryUnavailable("no finished sessions to start from")
        session = finished_sessions[-1]
    finished = views.last_finished_minute(session)
    if finished is None:
        raise SummaryUnavailable(f"no candles for {session}")
    state_point = NSE.epoch_of(date.fromisoformat(session), LAST_INTRADAY_POINT)
    as_of = at if at is not None else min(finished, state_point)
    rows = score_symbols(model, views, session, as_of, params)
    if rows.empty:
        raise SummaryUnavailable(f"no features for {session} at {NSE.time_of(as_of):%H:%M}")
    nd, known = next_trading_day(date.fromisoformat(session))
    latest = finished_sessions[-1] if finished_sessions else None
    # The screen that did best in testing: the widest range this session, so far.
    span: dict[str, list[float]] = {}
    for c in views.intraday_candles(session, as_of=as_of):
        r = span.setdefault(c.symbol, [c.high, c.low, c.close])
        r[0], r[1], r[2] = max(r[0], c.high), min(r[1], c.low), c.close
    widest = sorted(((h - l) / cl * 100.0, sym) for sym, (h, l, cl) in span.items() if cl)[::-1][:5]
    out = build_summary(rows, cfg, version, card, session, as_of, finished, latest,
                        found[1] if found else None, min_chance, top, live=session == today,
                        next_session={"date": nd.isoformat(), "known": known,
                                      "partial": as_of < state_point,
                                      "widest_range": [{"symbol": sym, "range_pct": rng} for rng, sym in widest]})
    return out


def pick_rows(s: dict[str, Any], made_at: Optional[int] = None) -> list[dict]:
    """A summary as rows for ``Store.save_summary_picks``: every stock it scored."""
    made_at = made_at or int(time.time())
    nxt = s.get("next_session")
    return [{
        "made_at": made_at, "kind": "next" if nxt else "today", "model_version": s["model_version"],
        "target_pct": s["target_pct"], "session_date": nxt["date"] if nxt else s["session"],
        "base_session": s["session"], "as_of": s["as_of"], "symbol": r["symbol"], "rank": r["rank"],
        "price": r["price"], "chance": r["chance"], "p_model": r["p_model"], "gain_p25": r["p25"],
        "gain_p50": r["p50"], "gain_p75": r["p75"], "passed": int(r["passed"]),
    } for r in s["all"]]
