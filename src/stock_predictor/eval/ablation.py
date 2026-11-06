"""Measure each candidate input on this project's own target (design 4.1).

The inputs were tested elsewhere against next-day moves. Here the question is
narrower and the target different: at prediction point T, does the input tell
anything about the **best window still open after T** -- the label every model
in this project learns?

For each input, over the given sessions:

* **coverage**  -- share of symbol-sessions where the value exists at all;
* **IC**        -- mean per-session Spearman correlation between the value and
  the symbol's net-of-cost best remaining window, with a 95% bootstrap interval
  over sessions. Near zero with an interval spanning zero = no information;
* **pass / fail** -- what the input's filter rule (thresholds from the params)
  does: mean net gain of symbols that pass minus those that fail, per session,
  and the adverse rate on each side;
* **top-k delta** -- change in a baseline's realised net gain when that one
  filter is applied to its ranking.

Then **score order**: symbols bucketed into quintiles by a score (the model's
when one is given, else 20-day ATR -- the volatility baseline's), with mean net
gain and adverse rate per quintile. If the top quintile's gain rises only with
its adverse rate, the score is ranking volatility, not quality.

Intervals are percentile bootstraps over *sessions* -- symbols on the same day
are not independent draws, sessions are closer to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from stock_predictor.config import CONFIG, OPENING_RANGE_T1
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.eval.backtest import as_of_for
from stock_predictor.eval.baselines import _atr_20, run_baseline
from stock_predictor.eval.charges import net_gain_pct
from stock_predictor.features.candidates import CANDIDATE_COLUMNS, candidate_values, is_missing
from stock_predictor.filters import evaluate_rule
from stock_predictor.labels.targets import adverse_excursions, build_targets
from stock_predictor.params import Params, get_params

TOPK_BASELINES = ("highest_volatility", "yesterdays_leaders")
ScoreFn = Callable[[AnalyzerViews, str, int], "dict[str, float]"]


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        return self.n > 1 and (self.lo > 0 or self.hi < 0)

    def fmt(self, digits: int = 3) -> str:
        if self.n == 0 or math.isnan(self.mean):
            return "n/a"
        return f"{self.mean:+.{digits}f} [{self.lo:+.{digits}f},{self.hi:+.{digits}f}]"


def bootstrap(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Interval:
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if len(v) == 1:
        return Interval(float(v[0]), float(v[0]), float(v[0]), 1)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return Interval(float(v.mean()), float(np.percentile(means, 2.5)),
                    float(np.percentile(means, 97.5)), len(v))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 5:
        return float("nan")
    rx = _ranks(x)
    ry = _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(a: Sequence[float]) -> np.ndarray:
    import pandas as pd

    return pd.Series(list(a)).rank(method="average").to_numpy()


@dataclass
class InputReport:
    name: str
    column: str
    feature_on: bool
    filter_on: bool
    rule: str
    coverage: float = 0.0
    ic: Optional[Interval] = None
    pass_rate: float = float("nan")
    gain_diff: Optional[Interval] = None
    adverse_pass: float = float("nan")
    adverse_fail: float = float("nan")
    topk_delta: dict[str, Interval] = field(default_factory=dict)


@dataclass
class QuintileRow:
    quintile: int            # 5 = highest score
    net_gain: float
    adverse_rate: float
    mean_score: float


@dataclass
class AblationReport:
    sessions: int
    as_of_label: str
    adverse_move_pct: float
    inputs: list[InputReport]
    score_label: str
    quintiles: list[QuintileRow]
    rank_by_compare: dict[str, dict[str, float]] = field(default_factory=dict)


def _rule_text(name: str, p: Params) -> str:
    return {
        "opening_check": f"at {p.opening_check.check_time}: range in "
                         f"{list(p.opening_check.allowed_range_states)}"
                         + (" and above VWAP" if p.opening_check.require_above_vwap else ""),
        "run_up": f"{p.run_up.lookback_days}d run-up <= {p.run_up.max_pct:g}%",
        "rsi": f"{p.rsi.min:g} <= RSI{p.rsi.period} <= {p.rsi.max:g}",
        "volume": f"rel. volume ({p.volume.lookback_days}d) >= {p.volume.min_relative_volume:g}",
        "close_location": f"{p.close_location.min:g} <= close location <= {p.close_location.max:g}",
        "momentum": f"within {p.momentum.max_dist_from_high_pct:g}% of "
                    f"{p.momentum.lookback_days}d high",
    }[name]


def _atr_scores(views: AnalyzerViews, session_date: str, as_of: int) -> dict[str, float]:
    universe = views.universe(session_date)
    daily = views.daily_candles(symbols=universe, before=session_date)
    out = {}
    for s in universe:
        v = _atr_20(daily, s)
        if v is not None:
            out[s] = v
    return out


def run_ablation(
    views: AnalyzerViews,
    sessions: Sequence[str],
    clock_time=OPENING_RANGE_T1,
    params: Optional[Params] = None,
    k: Optional[int] = None,
    score_fn: Optional[ScoreFn] = None,
    score_label: str = "20-day ATR %",
    model=None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    frames=None,
) -> AblationReport:
    """``frames`` is an optional ``FrameSource``: with a model, its saved
    session frames are cut to ``as_of`` instead of rebuilt."""
    params = params or get_params()
    k = k or CONFIG.top_n
    score_fn = score_fn or _atr_scores
    names = list(CANDIDATE_COLUMNS)
    primary = {n: CANDIDATE_COLUMNS[n][0] for n in names}

    ic: dict[str, list[float]] = {n: [] for n in names}
    diff: dict[str, list[float]] = {n: [] for n in names}
    n_avail: dict[str, int] = {n: 0 for n in names}
    n_pass: dict[str, int] = {n: 0 for n in names}
    adv: dict[str, dict[str, list[int]]] = {n: {"pass": [], "fail": []} for n in names}
    topk: dict[str, dict[str, list[float]]] = {n: {b: [] for b in TOPK_BASELINES} for n in names}
    total_symbol_sessions = 0
    quint_gain: list[list[float]] = [[] for _ in range(5)]
    quint_adv: list[list[float]] = [[] for _ in range(5)]
    quint_score: list[list[float]] = [[] for _ in range(5)]
    rank_cmp: dict[str, list[dict[str, float]]] = {"score": [], "risk_adjusted": []}
    thr = params.score_order.adverse_move_pct
    scored_sessions = 0

    for i, sd in enumerate(sessions, start=1):
        if progress:
            progress(i, len(sessions), sd)
        as_of = as_of_for(sd, clock_time)
        truth = build_targets(views, sd, as_of)
        if not truth:
            continue
        scored_sessions += 1
        universe = views.universe(sd)
        net = {s: 0.0 for s in universe}
        for t in truth:
            net[t.symbol] = net_gain_pct(t.gain_pct, entry_price=1000.0)
        adverse = adverse_excursions(views, sd, as_of)
        is_adverse = {s: int(a >= thr) for s, a in adverse.items()}
        values = candidate_values(views, sd, as_of, universe, params)
        total_symbol_sessions += len(universe)
        rankings = {b: [p.symbol for p in run_baseline(b, views, sd, as_of, 10**6)]
                    for b in TOPK_BASELINES}

        for name in names:
            col = primary[name]
            have = [s for s in universe if not is_missing(values.get(s, {}).get(col))]
            n_avail[name] += len(have)
            ic[name].append(_spearman([values[s][col] for s in have], [net[s] for s in have]))

            verdict = {s: evaluate_rule(name, values.get(s, {}), params) for s in universe}
            passed = [s for s, v in verdict.items() if v is True]
            failed = [s for s, v in verdict.items() if v is False]
            n_pass[name] += len(passed)
            if passed and failed:
                diff[name].append(float(np.mean([net[s] for s in passed]) -
                                        np.mean([net[s] for s in failed])))
            adv[name]["pass"].extend(is_adverse[s] for s in passed if s in is_adverse)
            adv[name]["fail"].extend(is_adverse[s] for s in failed if s in is_adverse)

            for b, order in rankings.items():
                if not order:
                    continue
                base = order[:k]
                filt = [s for s in order if verdict.get(s) is not False][:k]
                if not filt:
                    continue
                topk[name][b].append(float(np.mean([net[s] for s in filt]) -
                                           np.mean([net[s] for s in base])))

        scores = score_fn(views, sd, as_of)
        ranked = sorted((s for s in scores if s in net), key=lambda s: scores[s])
        if len(ranked) >= 5:
            for q, chunk in enumerate(np.array_split(np.array(ranked, dtype=object), 5)):
                chunk = list(chunk)
                quint_gain[q].append(float(np.mean([net[s] for s in chunk])))
                quint_adv[q].append(float(np.mean([is_adverse.get(s, 0) for s in chunk])))
                quint_score[q].append(float(np.mean([scores[s] for s in chunk])))

        if model is not None:
            from dataclasses import replace

            from stock_predictor.features.pipeline import build_feature_frame
            from stock_predictor.models.rank import rank_session

            if frames is not None:
                frame = frames.at(sd, as_of)
            else:
                frame = build_feature_frame(views, sd, as_of, params=params)
            for rb in rank_cmp:
                p_rb = replace(params, score_order=replace(params.score_order, rank_by=rb))
                top = rank_session(model, views, sd, as_of, k, p_rb, frame=frame)
                if top:
                    rank_cmp[rb].append({
                        "realised_net_gain": float(np.mean([net.get(t.symbol, 0.0) for t in top])),
                        "adverse_rate": float(np.mean([is_adverse.get(t.symbol, 0) for t in top])),
                    })

    reports = []
    for name in names:
        sec = params.candidates()[name]
        verdicts = n_avail[name]
        reports.append(InputReport(
            name=name, column=primary[name], feature_on=sec.feature, filter_on=sec.filter,
            rule=_rule_text(name, params),
            coverage=n_avail[name] / total_symbol_sessions if total_symbol_sessions else 0.0,
            ic=bootstrap(ic[name]),
            pass_rate=n_pass[name] / verdicts if verdicts else float("nan"),
            gain_diff=bootstrap(diff[name]),
            adverse_pass=float(np.mean(adv[name]["pass"])) if adv[name]["pass"] else float("nan"),
            adverse_fail=float(np.mean(adv[name]["fail"])) if adv[name]["fail"] else float("nan"),
            topk_delta={b: bootstrap(v) for b, v in topk[name].items()},
        ))

    quintiles = [
        QuintileRow(q + 1, float(np.mean(quint_gain[q])) if quint_gain[q] else float("nan"),
                    float(np.mean(quint_adv[q])) if quint_adv[q] else float("nan"),
                    float(np.mean(quint_score[q])) if quint_score[q] else float("nan"))
        for q in range(5)
    ]
    compare = {
        rb: {m: float(np.mean([r[m] for r in rows])) for m in ("realised_net_gain", "adverse_rate")}
        for rb, rows in rank_cmp.items() if rows
    }
    return AblationReport(
        sessions=scored_sessions,
        as_of_label=clock_time.strftime("%H:%M"),
        adverse_move_pct=thr,
        inputs=reports,
        score_label=score_label,
        quintiles=quintiles,
        rank_by_compare=compare,
    )


def format_report(r: AblationReport) -> str:
    lines = [
        f"candidate inputs at {r.as_of_label} over {r.sessions} sessions",
        "target: net-of-cost best window still open after T; intervals are 95% "
        "bootstraps over sessions",
        "",
        f"{'input':<15}{'sw':<5}{'cover':>6}  {'IC vs remaining gain':<24}{'pass':>6}  "
        f"{'gain pass-fail (pp)':<24}{'adverse p/f':>12}",
        "-" * 98,
    ]
    for x in r.inputs:
        sw = ("F" if x.feature_on else "-") + ("X" if x.filter_on else "-")
        adv = "n/a" if math.isnan(x.adverse_pass) or math.isnan(x.adverse_fail) else \
            f"{x.adverse_pass:.0%}/{x.adverse_fail:.0%}"
        pr = "n/a" if math.isnan(x.pass_rate) else f"{x.pass_rate:.0%}"
        lines.append(
            f"{x.name:<15}{sw:<5}{x.coverage:>6.0%}  {x.ic.fmt():<24}{pr:>6}  "
            f"{x.gain_diff.fmt():<24}{adv:>12}"
        )
    lines += ["", "rules (from params):"]
    lines += [f"  {x.name:<15}{x.rule}" for x in r.inputs]
    lines += ["", "change in a baseline's top-k realised net gain (pp) with that one filter applied:",
              f"{'input':<15}" + "".join(f"{b:<30}" for b in TOPK_BASELINES)]
    for x in r.inputs:
        lines.append(f"{x.name:<15}" + "".join(f"{x.topk_delta[b].fmt():<30}" for b in TOPK_BASELINES))
    lines += ["", f"score order -- quintiles by {r.score_label} (5 = highest); "
                  f"adverse = fell >= {r.adverse_move_pct:g}% after T:",
              f"{'quintile':<10}{'mean score':>12}{'net gain %':>12}{'adverse':>10}"]
    for q in r.quintiles:
        lines.append(f"{q.quintile:<10}{q.mean_score:>12.3f}{q.net_gain:>12.3f}{q.adverse_rate:>10.0%}")
    if r.rank_by_compare:
        lines += ["", "model top-k by rank_by:",
                  f"{'rank_by':<16}{'net gain %':>12}{'adverse':>10}"]
        for rb, m in r.rank_by_compare.items():
            lines.append(f"{rb:<16}{m['realised_net_gain']:>12.3f}{m['adverse_rate']:>10.0%}")
    lines += ["", "sw: F = feature on, X = filter on (params). IC/pass-fail intervals that span "
                  "zero mean no measurable effect at this sample size."]
    return "\n".join(lines)
