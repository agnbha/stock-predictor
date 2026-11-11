"""The live board behind ``/dashboard``: which symbols are likely now, how
sure, why, and what stock-analyzer is doing.

A background thread refreshes it; requests only read the last snapshot, so the
model never runs on the request path (design section 7).

For every symbol, at the latest minute most symbols have finished:

* **Likelihood**: the fixed-target model's P(target), and that probability
  mapped to how often it came true on sessions the model had not trained on
  (the calibration kept on its card), with how many cases back that rate.
* **Why**: the inputs that moved this symbol's probability most. Each input
  (with its cross-sectional rank and z-score) is set to today's median across
  the universe and the probability recomputed; the difference is the input's
  effect, in percentage points of the model's P.
* **Screener**: stock-analyzer's fundamentals, latest CANSLIM score, bullish
  score and watchlist, and for each signal what ``screener-evidence``
  measured it to be worth on this target.
* **Conclusion**: those three in a few sentences, every figure in them shown
  elsewhere on the board.

After 14:49 a 40-minute window no longer fits in the session, and the model was
never trained on such rows, so the board holds the last minute that fits.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np

from stock_predictor.clock import NSE
from stock_predictor.config import CONFIG
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.data.screener import ScreenerData
from stock_predictor.eval.screener import DEFINITIONS, letters_passed

log = logging.getLogger(__name__)

# Readable names for the model's inputs. Cross-sectional variants
# (``x__xs_rank``, ``x__xs_z``) are grouped with their base input.
FEATURE_LABELS = {
    "minutes_since_open": "minutes since the open",
    "session_progress_sin": "time of day",
    "session_progress_cos": "time of day",
    "day_of_week": "day of week",
    "ret_1": "return, last minute",
    "ret_5": "return, last 5 minutes",
    "ret_15": "return, last 15 minutes",
    "ret_30": "return, last 30 minutes",
    "ret_60": "return, last 60 minutes",
    "volatility_30": "volatility, last 30 minutes",
    "atr_14": "14-minute ATR",
    "body_ratio": "last candle's body",
    "upper_wick_ratio": "last candle's upper wick",
    "lower_wick_ratio": "last candle's lower wick",
    "volume_z_time_of_day": "volume vs usual for this minute",
    "dist_from_vwap_pct": "distance from VWAP",
    "dist_from_open_range_high_pct": "distance from opening-range high",
    "dist_from_open_range_low_pct": "distance from opening-range low",
    "dist_from_prior_close_pct": "change since yesterday's close",
    "dist_from_prior_high_pct": "distance from yesterday's high",
    "position_in_day_range": "position in today's range",
    "gap_pct": "opening gap",
    "event_opening_range_breakout": "opening-range breakout",
    "event_vwap_reclaim": "VWAP reclaim",
    "event_volume_spike": "volume spike",
    "event_prior_day_high_break": "break of yesterday's high",
    "dist_from_sma5_pct": "distance from 5-day average",
    "dist_from_sma20_pct": "distance from 20-day average",
    "dist_from_sma50_pct": "distance from 50-day average",
    "rsi14": "daily RSI (14)",
    "trend_score": "daily trend score",
    "pct_of_10d_range": "position in 10-day range",
    "beta": "beta to the universe",
    "regime_advancing_pct": "market: share advancing yesterday",
    "regime_mean_change_pct": "market: mean change yesterday",
    "regime_dispersion_pct": "market: dispersion yesterday",
    "regime_minutes_since_open": "minutes since the open",
    "regime_day_of_week": "day of week",
}
# Inputs that are the same for every symbol at a minute move nothing between
# symbols, so they are left out of "why".
SHARED = {"minutes_since_open", "session_progress_sin", "session_progress_cos", "day_of_week",
          "regime_advancing_pct", "regime_mean_change_pct", "regime_dispersion_pct",
          "regime_minutes_since_open", "regime_day_of_week"}
PERCENT = {"ret_1", "ret_5", "ret_15", "ret_30", "ret_60", "dist_from_vwap_pct",
           "dist_from_open_range_high_pct", "dist_from_open_range_low_pct",
           "dist_from_prior_close_pct", "dist_from_prior_high_pct", "gap_pct",
           "dist_from_sma5_pct", "dist_from_sma20_pct", "dist_from_sma50_pct"}


def _family(name: str) -> str:
    return name.split("__")[0]


def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) or np.isinf(f) else f


def analyzer_config(db_path: Path) -> dict:
    """stock-analyzer's model-client and monitor settings, from its
    application.properties beside the database (an environment variable such
    as MODEL_ENABLED on its daemon would override what is read here)."""
    props = db_path.parent.parent / "src" / "main" / "resources" / "application.properties"
    out: dict[str, Any] = {"source": str(props), "found": props.exists()}
    if not props.exists():
        return out
    kv = {}
    for line in props.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    out.update({
        "model_enabled": kv.get("model.enabled", "false").lower() == "true",
        "model_url": kv.get("model.service.url"),
        "model_horizon_minutes": _num(kv.get("model.horizon.minutes")),
        "poll_seconds": _num(kv.get("monitor.poll.interval.seconds")),
    })
    return out


def drivers(model, rows, top_up: int = 4, top_down: int = 2) -> list[list[dict]]:
    """Per row: the inputs whose replacement by today's cross-sectional median
    moves the model's P most, as {feature, label, value, median, effect_pp}."""
    names = list(model.feature_names)
    X = rows[names].to_numpy(dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    base = np.asarray(model._clf.predict_proba(X))
    fams: dict[str, list[int]] = {}
    for j, n in enumerate(names):
        if _family(n) not in SHARED:
            fams.setdefault(_family(n), []).append(j)
    keys = list(fams)
    stacked = np.repeat(X[None, :, :], len(keys), axis=0)  # family x row x feature
    for k, fam in enumerate(keys):
        stacked[k][:, fams[fam]] = med[fams[fam]]
    p = np.asarray(model._clf.predict_proba(stacked.reshape(-1, X.shape[1]))).reshape(len(keys), -1)
    effect = (base[None, :] - p) * 100.0  # family x row, percentage points
    out = []
    for i in range(X.shape[0]):
        order = np.argsort(-effect[:, i])
        picks = [k for k in order[:top_up] if effect[k, i] > 0.05]
        picks += [k for k in order[::-1][:top_down] if effect[k, i] < -0.05]
        row = []
        for k in picks:
            fam = keys[k]
            j = names.index(fam) if fam in names else fams[fam][0]
            value, median = _num(X[i, j]), _num(med[j])
            if fam == "atr_14":  # as % of price, which is how it reads
                close = _num(rows["_close"].iloc[i]) or 0
                value = value / close * 100 if value is not None and close else value
                median = None
            row.append({"feature": fam, "label": FEATURE_LABELS.get(fam, fam), "value": value,
                        "median": median, "percent": fam in PERCENT or fam == "atr_14",
                        "effect_pp": float(effect[k, i])})
        out.append(row)
    return out


class LiveBoard:
    """``views`` must be the board's own (``live=True``): the board reads it from
    its thread, and an SQLite connection used by two threads at once slows to
    a crawl."""

    def __init__(self, views: AnalyzerViews, service=None, model=None, card=None,
                 version: Optional[str] = None, evaluation: Optional[dict] = None,
                 evaluation_version: Optional[str] = None, evidence: Optional[dict] = None,
                 evidence_version: Optional[str] = None, refresh_seconds: float = 20.0):
        self.views = views
        self.screener = ScreenerData(views)
        self.service = service
        self.model, self.card, self.version = model, card, version
        self.evaluation, self.evaluation_version = evaluation, evaluation_version
        self.evidence, self.evidence_version = evidence, evidence_version
        self.refresh_seconds = refresh_seconds
        self.config = analyzer_config(Path(views.db_path))
        self._lock = threading.RLock()
        self._snapshot: dict = {"status": "starting"}
        self._candidates: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cfg = None
        if card is not None:
            from stock_predictor.models import card_target

            self.cfg = card_target(card)

    # -- thread ---------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-board")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception:  # pragma: no cover - keep serving the last good board
                log.exception("live board refresh failed")
            self._stop.wait(self.refresh_seconds)

    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self._snapshot)
        snap["requests"] = self._requests()
        snap["ranking"] = self._ranking()
        snap["served_at"] = int(time.time())
        return snap

    # -- one refresh ------------------------------------------------------------

    def refresh(self) -> dict:
        now = int(time.time())
        session = self.views.session_in_progress() or NSE.session_date_of(now).isoformat()
        day = date.fromisoformat(session)
        open_ep, close_ep = NSE.session_open_epoch(day), NSE.session_close_epoch(day)
        finished = self.views.last_finished_minute(session)
        market = {
            "session": session, "now": now, "open": open_ep, "close": close_ep,
            "state": "before the open" if now < open_ep else ("open" if now < close_ep else "closed"),
            "last_finished_minute": finished,
            "lag_seconds": (now - finished - 60) if finished else None,
        }
        if self.model is not None and finished is not None:
            if self.cfg.to_close:  # every minute's window ends at the close
                last_fit = close_ep - 60
            else:
                last_fit = close_ep - 60 - self.cfg.horizon_minutes * 60
            as_of = min(finished, last_fit)
            if self._candidates is None or self._candidates["as_of"] != as_of:
                self._candidates = self._score(session, as_of)
            market["as_of"] = as_of
            market["held"] = finished > last_fit
            market["window_end"] = close_ep - 60 if self.cfg.to_close else as_of + self.cfg.horizon_minutes * 60
        snap = {
            "status": "ok",
            "refreshed_at": now,
            "market": market,
            "model": self._model_info(),
            "candidates": (self._candidates or {}).get("rows", []),
            "volatility_top": (self._candidates or {}).get("volatility_top", []),
            "evidence": self._evidence_rows(),
            "analyzer": self._analyzer(session, finished),
        }
        if self.model is None:
            snap["status"] = "no fixed-target model: run `stock-predictor train --target 1 --horizon 40`"
        elif finished is None:
            snap["status"] = f"no candles for {session} yet"
        with self._lock:
            self._snapshot = snap
        return snap

    def _score(self, session: str, as_of: int) -> dict:
        from stock_predictor.features.pipeline import build_feature_frame, latest_row_per_symbol
        from stock_predictor.params import Params

        params = Params.from_dict(self.card.params) if self.card.params else None
        frame = build_feature_frame(self.views, session, as_of, params=params)
        if frame.empty:
            return {"as_of": as_of, "rows": [], "volatility_top": []}
        rows = latest_row_per_symbol(frame).reset_index(drop=True)
        pred = self.model.predict(rows)
        why = drivers(self.model, rows)
        vol = (rows["atr_14"] / rows["_close"].replace(0, np.nan) * 100).to_numpy()
        vol_rank = (-np.nan_to_num(vol, nan=-1)).argsort().argsort() + 1
        screener = self._screener_now(session, list(rows["symbol"]))
        base = self._base_rate(as_of)
        cfg = self.cfg
        out = []
        for i, r in rows.iterrows():
            p = float(pred.p_up[i])
            seen = float(pred.p_seen[i]) if pred.p_seen is not None else None
            band = self._band(p)
            close = float(r["_close"])
            item = {
                "symbol": r["symbol"], "last_ts": int(r["ts_epoch"]), "price": close,
                "target_price": close * (1 + cfg.target_pct / 100),
                "stop_price": close * (1 - cfg.stop_pct / 100) if cfg.stop_pct is not None else None,
                "day_change_pct": _num(r.get("dist_from_prior_close_pct")),
                "p_model": p, "p_seen": seen, "band": band,
                "return_p25": float(pred.gain_p25[i]), "return_p50": float(pred.gain_p50[i]),
                "return_p75": float(pred.gain_p75[i]),
                "vol_pct": _num(vol[i]), "vol_rank": int(vol_rank[i]),
                "drivers": why[i], "screener": screener.get(r["symbol"], {}),
            }
            out.append(item)
        key = "p_seen" if out and out[0]["p_seen"] is not None else "p_model"
        out.sort(key=lambda x: -x[key])
        for rank, item in enumerate(out, start=1):
            item["rank"] = rank
            item["conclusion"] = self._conclusion(item, base, len(out))
        vt = sorted(out, key=lambda x: x["vol_rank"])[:5]
        return {"as_of": as_of, "rows": out,
                "volatility_top": [{"symbol": x["symbol"], "vol_pct": x["vol_pct"], "rank": x["rank"]} for x in vt]}

    # -- pieces ---------------------------------------------------------------

    def _band(self, p: float) -> Optional[dict]:
        for b in (self.card.calibration if self.card else []) or []:
            lo, hi = (float(x) for x in b["band"].split("-"))
            if (lo < p <= hi) or (lo == 0 and p <= hi):
                return b
        return None

    def _base_rate(self, as_of: int) -> Optional[dict]:
        """The held-out evaluation's row for the nearest point at or before as_of."""
        if not self.evaluation:
            return None
        hhmm = NSE.time_of(as_of).strftime("%H:%M")
        rows = [r for r in self.evaluation.get("by_point", []) if r["point"] <= hhmm]
        return rows[-1] if rows else None

    def _screener_now(self, session: str, symbols: list[str]) -> dict[str, dict]:
        funds = self.screener.company_fundamentals()
        cans = self.screener.canslim_latest()
        bull: dict[str, dict] = {}
        for r in self.screener.bullish_history():
            if r["session_date"] < session:
                bull[r["symbol"]] = r  # ordered by session: the last one wins
        scores = sorted((v["bullish_score"] for k, v in bull.items() if k in symbols), reverse=True)
        cut = scores[max(0, int(len(scores) * 0.2) - 1)] if scores else None
        picks = self.screener.watchlist_picks()
        prior = sorted({p["as_of_date"] for p in picks if p["as_of_date"] < session})
        wl = {p["symbol"]: p for p in picks if prior and p["as_of_date"] == prior[-1]}
        out = {}
        for s in symbols:
            f, c, b = funds.get(s, {}), cans.get(s, {}), bull.get(s)
            out[s] = {
                "fundamentals": {k: f.get(k) for k in (
                    "pe_now", "pb_now", "roe", "roce", "debt_to_equity", "sales_growth_ttm",
                    "profit_growth_ttm", "profit_growth_3y", "market_cap_now_cr", "pct_from_52w_high",
                    "results_quarter", "classification")} if f else None,
                "canslim": {"as_of": c.get("as_of"), "letters": c.get("letters"),
                            "letters_passed": letters_passed(c.get("letters")) if c else None,
                            "rs_rating": c.get("rs_rating"), "signal": c.get("signal"),
                            "reason": c.get("reason"), "pct_from_pivot": c.get("pct_from_pivot"),
                            "market_state": c.get("market_state")} if c else None,
                "bullish": {"session": b["session_date"], "score": b["bullish_score"],
                            "rank": scores.index(b["bullish_score"]) + 1 if b["bullish_score"] in scores else None,
                            "of": len(scores)} if b else None,
                "watchlist": {"as_of": prior[-1], "rank": wl[s]["rank"]} if s in wl else None,
                "groups": {
                    "bullish_top_fifth": bool(b and cut is not None and b["bullish_score"] >= cut),
                    "watchlist_pick": s in wl,
                    "canslim_rs_80": bool(c and (c.get("rs_rating") or 0) >= 80),
                    "canslim_4_letters": bool(c and letters_passed(c.get("letters")) >= 4),
                },
            }
        return out

    def _evidence_rows(self) -> Optional[dict]:
        if not self.evidence:
            return None
        return {"version": self.evidence_version, "sessions": self.evidence.get("sessions"),
                "rows": self.evidence.get("evidence", [])}

    def _conclusion(self, item: dict, base: Optional[dict], universe: int) -> list[str]:
        lines = []
        p = item["p_seen"] if item["p_seen"] is not None else item["p_model"]
        band = item["band"]
        cfg = self.cfg
        s = f"About {p:.0%} chance of touching {item['target_price']:.2f} (+{cfg.target_pct:g}%) by the end of the window"
        if band:
            s += (f": the model said {band['predicted']:.0%} on {band['rows']:,} cases it had not trained on, "
                  f"and {band['seen']:.0%} came true")
        lines.append(s + ".")
        if base:
            lines.append(f"At this time of day, {base['base_hit']:.0%} of stocks did it in the held-out month; "
                         f"the model's top 5 {base['model_hit']:.0%}, the 5 most volatile {base['intraday_volatility_hit']:.0%}.")
        if item["vol_rank"] <= 5:
            lines.append(f"Also among the 5 most volatile stocks now (#{item['vol_rank']}), the simple screen the model has not beaten.")
        ev = {e["signal"]: e for e in (self.evidence or {}).get("evidence", [])}
        for sig, on in (item["screener"].get("groups") or {}).items():
            if not on:
                continue
            e = ev.get(sig)
            what = DEFINITIONS[sig]
            if e is None or e.get("lift") is None:
                lines.append(f"Screener: {what}; not measured yet (`stock-predictor screener-evidence`).")
            elif e["reading"].startswith("too little") and (e.get("top_reading") or "").startswith("too little"):
                lines.append(f"Screener: {what}. Only {e['sessions']} sessions of history "
                             f"({e['group_symbol_days']:,} symbol-days): too little to tell.")
            else:
                s = (f"Screener: {what}. On its own that has meant {e['group_hit']:.1%} vs {e['rest_hit']:.1%} "
                     f"({e['lift']:.2f}x, {e['group_symbol_days']:,} symbol-days): {e['reading']}")
                if e.get("top_hit") is not None and e.get("top_rest_hit") is not None:
                    s += (f". Among the model's top 5 it has meant {e['top_hit']:.0%} vs {e['top_rest_hit']:.0%} "
                          f"({e.get('top_symbol_days', 0):,} symbol-days): "
                          + {"supports": "it adds to the model's call",
                             "against": "it counts against the model's call",
                             "no measurable difference": "the model already sees it",
                             }.get(e.get("top_reading"), "too little history to tell"))
                lines.append(s + ".")
        c = item["screener"].get("canslim") or {}
        if c.get("signal") and c["signal"] != "NONE":
            lines.append(f"CANSLIM signal {c['signal']}: {c.get('reason') or ''}".strip())
        return lines

    def _model_info(self) -> Optional[dict]:
        if self.card is None:
            return None
        c = self.card
        info = {
            "version": self.version, "target": self.cfg.describe(), "target_cfg": asdict(self.cfg),
            "trained_through": c.trained_through, "train_sessions": c.train_sessions,
            "minute_step": c.minute_step, "features": len(c.feature_names), "backend": c.backend,
            "candidate_features": [k for k, v in (c.params or {}).items() if isinstance(v, dict) and v.get("feature")],
            "calibration": c.calibration,
        }
        if self.evaluation:
            e = self.evaluation
            info["evaluation"] = {"version": self.evaluation_version, "sessions": e.get("sessions"),
                                  "session_count": e.get("session_count"), "top_n": e.get("top_n"),
                                  "cost_pct": e.get("cost_pct"), "summary": e.get("summary")}
        return info

    def _analyzer(self, session: str, finished: Optional[int]) -> dict:
        live = self.views._live_candles(session) or []
        latest: dict[str, int] = {}
        for cnd in live:
            latest[cnd.symbol] = max(latest.get(cnd.symbol, 0), cnd.ts_epoch)
        tracked = [{"symbol": s, "last_ts": t,
                    "behind_minutes": int((finished - t) // 60) if finished else None}
                   for s, t in sorted(latest.items())]
        trades = self.screener.trades(session)
        bought = sorted({t["symbol"] for t in trades if t["side"] == "BUY"})
        return {
            "config": self.config,
            "heartbeat": self.screener.monitor_heartbeat(),
            "tracked": tracked,
            "bought_today": bought,
            "untracked_buys": [s for s in bought if s not in latest],
            "trades_today": trades,
            "holdings": self.screener.holdings(),
        }

    def _requests(self) -> dict:
        if self.service is None:
            return {"total": 0, "by_symbol": []}
        return self.service.requests.summary()

    def _ranking(self) -> Optional[dict]:
        if self.service is None:
            return None
        r = self.service.cache.get()
        if r is None:
            return None
        return {"as_of": r.as_of, "computed_at": int(r.computed_at), "session": r.session_date,
                "model_version": r.model_version,
                "picks": [asdict(p) for p in r.picks]}
