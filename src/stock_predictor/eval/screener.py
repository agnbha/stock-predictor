"""Does stock-analyzer's screener data say anything about a fixed target?

For each screener signal that has history, every symbol at every prediction
point is put in or out of a group, using only what was published before that
session's open, and the target's hit rate is compared between the two. When a
model is given, the same comparison is made among the model's top picks on
sessions it was not trained on: the question there is whether the screener
should change the model's call.

* **Bullish score** (``v_bullish_history``): the previous session's score,
  top fifth of the universe that day.
* **Watchlist pick** (``watchlist_pick``): on stock-analyzer's list made after
  the previous session.
* **CANSLIM** (``v_canslim_signal_history``): the latest run finished before
  the open. RS rating 80 or more; four or more of the seven letters passed.
  Runs exist only since 2026-10-26.

The fundamentals (P/E, ROE, growth) are one snapshot per company, so there is
nothing point-in-time to measure; the dashboard shows them as context only.

Rows at neighbouring points share most of a window, so the counts that bound
how much to trust a figure are the symbol-days, reported beside every rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np

from stock_predictor.clock import NSE
from stock_predictor.data.screener import ScreenerData
from stock_predictor.eval.target import default_points, point_rows

# Below this many symbol-days in a group, a difference is not read either way.
MIN_SYMBOL_DAYS = 100
SUPPORTS, AGAINST = 1.2, 1 / 1.2


@dataclass
class Evidence:
    signal: str
    definition: str
    sessions: int
    group_rows: int
    group_symbol_days: int
    group_hit: Optional[float]
    rest_hit: Optional[float]
    lift: Optional[float]
    top_rows: int = 0               # model's top picks in the group, out of sample
    top_hit: Optional[float] = None
    top_rest_rows: int = 0
    top_rest_hit: Optional[float] = None
    top_symbol_days: int = 0
    reading: str = ""       # on its own: in the group vs the rest
    top_reading: str = ""   # among the model's top picks: does it change the call?

    def to_dict(self) -> dict:
        return asdict(self)


def reading(lift: Optional[float], symbol_days: int) -> str:
    if lift is None or symbol_days < MIN_SYMBOL_DAYS:
        return "too little history to tell"
    if lift >= SUPPORTS:
        return "supports"
    if lift <= AGAINST:
        return "against"
    return "no measurable difference"


def letters_passed(letters: Optional[str]) -> int:
    """CANSLIM letters string, e.g. '-A-SLI-': a letter is a pass, '-' a fail, '?' unknown."""
    return sum(ch.isalpha() for ch in (letters or ""))


def screener_groups(screener: ScreenerData, keys):
    """For (session_date, symbol) keys, each signal's membership, point-in-time.
    Returns a DataFrame indexed like ``keys`` with one boolean column per signal
    (NaN where the signal has no value yet)."""
    import pandas as pd

    k = keys[["session_date", "symbol"]].drop_duplicates().copy()
    k["sd"] = pd.to_datetime(k["session_date"])
    k = k.sort_values("sd")

    out = k[["session_date", "symbol"]].copy()

    bh = pd.DataFrame(screener.bullish_history())
    if not bh.empty:
        bh["sd"] = pd.to_datetime(bh["session_date"])
        m = pd.merge_asof(k, bh[["sd", "symbol", "bullish_score"]].sort_values("sd"),
                          on="sd", by="symbol", allow_exact_matches=False)
        rank = m.groupby("session_date")["bullish_score"].rank(pct=True)
        out["bullish_top_fifth"] = np.where(m["bullish_score"].isna(), np.nan, (rank > 0.8).astype(float))

    wl = pd.DataFrame(screener.watchlist_picks())
    if not wl.empty:
        dates = sorted(wl["as_of_date"].unique())
        picked = wl.groupby("as_of_date")["symbol"].agg(set).to_dict()

        def on_list(sd, sym):
            prior = [d for d in dates if d < sd]
            return float(sym in picked[prior[-1]]) if prior else np.nan

        out["watchlist_pick"] = [on_list(sd, s) for sd, s in zip(out["session_date"], out["symbol"])]

    cs = pd.DataFrame(screener.canslim_history())
    if not cs.empty:
        cs = cs.sort_values("run_at")
        k2 = k.copy()
        k2["open_epoch"] = [NSE.session_open_epoch(date.fromisoformat(s)) for s in k2["session_date"]]
        k2 = k2.sort_values("open_epoch")
        m = pd.merge_asof(k2, cs[["run_at", "symbol", "rs_rating", "letters"]].rename(columns={"run_at": "open_epoch"}),
                          on="open_epoch", by="symbol", allow_exact_matches=False)
        m = m.set_index(["session_date", "symbol"]).reindex(pd.MultiIndex.from_frame(out[["session_date", "symbol"]]))
        rs = m["rs_rating"].to_numpy(dtype=float)
        out["canslim_rs_80"] = np.where(np.isnan(rs), np.nan, (rs >= 80).astype(float))
        lp = np.array([np.nan if not isinstance(x, str) else letters_passed(x) for x in m["letters"]])
        out["canslim_4_letters"] = np.where(np.isnan(lp), np.nan, (lp >= 4).astype(float))
    return out


DEFINITIONS = {
    "bullish_top_fifth": "bullish score (previous session) in the top fifth of the universe",
    "watchlist_pick": "on stock-analyzer's watchlist made after the previous session",
    "canslim_rs_80": "CANSLIM relative-strength rating 80 or more (latest run before the open)",
    "canslim_4_letters": "4 or more of the 7 CANSLIM letters passed (latest run before the open)",
}


def measure_screener(views, source, sessions: Sequence[str], model=None,
                     trained_through: Optional[str] = None, n: int = 5) -> list[Evidence]:
    """Evidence for each screener signal on ``source.cfg``'s target."""
    rows = point_rows(source, sessions, default_points(source.cfg))
    if rows.empty:
        return []
    rows["hit"] = (rows["label_class"] > 0).to_numpy()
    groups = screener_groups(ScreenerData(views), rows)
    rows = rows.merge(groups, on=["session_date", "symbol"], how="left")
    top = None
    if model is not None:
        oos = rows[rows["session_date"] > (trained_through or "")].copy()
        if not oos.empty:
            oos["p"] = model.predict(oos).p_up
            top = oos.sort_values("p", ascending=False, kind="stable").groupby(["session_date", "point"]).head(n)

    out = []
    for sig, definition in DEFINITIONS.items():
        if sig not in rows:
            continue
        have = rows[rows[sig].notna()]
        g, r = have[have[sig] == 1], have[have[sig] == 0]
        gh = float(g["hit"].mean()) if len(g) else None
        rh = float(r["hit"].mean()) if len(r) else None
        days = int(g[["session_date", "symbol"]].drop_duplicates().shape[0])
        lift = gh / rh if gh is not None and rh else None
        ev = Evidence(sig, definition, int(have["session_date"].nunique()), len(g), days, gh, rh, lift)
        if top is not None and sig in top:
            t = top[top[sig].notna()]
            tg, tr = t[t[sig] == 1], t[t[sig] == 0]
            ev.top_rows, ev.top_rest_rows = len(tg), len(tr)
            ev.top_hit = float(tg["hit"].mean()) if len(tg) else None
            ev.top_rest_hit = float(tr["hit"].mean()) if len(tr) else None
            ev.top_symbol_days = int(tg[["session_date", "symbol"]].drop_duplicates().shape[0])
            top_lift = ev.top_hit / ev.top_rest_hit if ev.top_hit is not None and ev.top_rest_hit else None
            ev.top_reading = reading(top_lift, ev.top_symbol_days)
        ev.reading = reading(lift, days)
        out.append(ev)
    return out
