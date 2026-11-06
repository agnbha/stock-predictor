"""``stock-predictor`` command line.

    build-features   build the feature frame for one session / prediction point
    train            fit the stage-1 model on the development sessions
                     (--target 1 --horizon 40: P(touch +1% within 40 minutes))
    predict          score the session in progress, now, with a trained model
    evaluate         baselines (and the model, if trained) over a prediction point
    evaluate-target  how often a fixed-target model's top picks reach the target
    backtest         evaluate across purged walk-forward folds
    ablate           measure each candidate screening input (design 4.1)
    params           print the effective parameters
    frames           show the saved training frames and their disk use
    serve            run the REST service + recompute scheduler
    score            fill realised gains on past predictions

Parameters for the candidate inputs come from --params FILE (TOML or JSON) or
$STOCK_PREDICTOR_PARAMS, then any --set section.key=value; see
params.example.toml.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from stock_predictor.clock import NSE
from stock_predictor.config import CONFIG, OPENING_RANGE_T1, PREOPEN_T0
from stock_predictor.data.analyzer import AnalyzerViews
from stock_predictor.data.store import Store
from stock_predictor.params import get_params, load_params, parse_set, set_params


def _clock_time(name: str, at: str = None):
    """``--at HH:MM`` wins over ``--point``."""
    from datetime import time

    from stock_predictor.config import LAST_INTRADAY_POINT

    if at:
        hh, mm = at.split(":")
        return time(int(hh), int(mm))
    return {"T0": PREOPEN_T0, "T1": OPENING_RANGE_T1, "Tn": LAST_INTRADAY_POINT}.get(name, OPENING_RANGE_T1)


def _point_label(args) -> str:
    return getattr(args, "at", None) or args.point


def _sessions(views: AnalyzerViews, args) -> list[str]:
    # Only sessions with final ground truth: never today's in-progress session,
    # never one the detector has not been run over.
    s = views.completed_sessions(start=getattr(args, "start", None), end=getattr(args, "end", None))
    from stock_predictor.eval.splits import holdout_split

    dev, hold = holdout_split(s, CONFIG.holdout_sessions)
    if getattr(args, "all_sessions", False):
        chosen = s
    else:
        chosen = hold if getattr(args, "holdout", False) else dev
    if getattr(args, "max_sessions", None):
        chosen = chosen[-args.max_sessions :]
    return chosen


def _frame_source(views: AnalyzerViews, args, params=None, cfg=None, sessions=None):
    """Saved session frames (design: docs/ARCHITECTURE.md), unless
    --no-frame-cache asks for every frame to be rebuilt and none saved.
    With ``sessions``, missing frames are built first in --jobs processes."""
    from stock_predictor.data.framestore import FrameStore
    from stock_predictor.models.training import FrameSource

    store = None if getattr(args, "no_frame_cache", False) else FrameStore()
    source = FrameSource(views, params or get_params(), cfg, store=store)
    jobs = getattr(args, "jobs", 1) or 1
    if sessions and store is not None and jobs > 1:
        built = source.prefetch(sessions, jobs)
        if built:
            print(f"built {built} missing session frames in {jobs} processes")
    return source


def _target_cfg(args):
    from stock_predictor.models.triple_barrier import TripleBarrierConfig

    if args.stop is not None and args.target is None:
        raise SystemExit("--stop needs --target")
    return TripleBarrierConfig(
        horizon_minutes=args.horizon or 30, target_pct=args.target, stop_pct=args.stop,
    )


def _load_model(name: str):
    from stock_predictor.models import ModelRegistry

    reg = ModelRegistry()
    version = name if name != "latest" else reg.latest()
    if not version:
        raise SystemExit("no trained model; run `stock-predictor train` first")
    model, card = reg.load(version)
    return version, model, card


def _card_params(card):
    from stock_predictor.params import Params

    return Params.from_dict(card.params) if card.params else get_params()


def _progress(label: str):
    def report(i, n, sd):
        print(f"\r  {label} {i}/{n} {sd}", end="" if i < n else "\n", file=sys.stderr, flush=True)
    return report


def _frames_line(source) -> str:
    if source.store is None:
        return f"frames: {source.built} built (cache off)"
    return f"frames: {source.loaded} read from {source.store.root}, {source.built} built and saved"


# -- commands -----------------------------------------------------------------


def cmd_build_features(args) -> int:
    from stock_predictor.features.pipeline import build_feature_frame

    with AnalyzerViews() as views:
        as_of = NSE.epoch_of(date.fromisoformat(args.session), _clock_time(args.point))
        frame = build_feature_frame(views, args.session, as_of, params=get_params())
    print(f"session={args.session} point={args.point} as_of={as_of}")
    print(f"rows={len(frame)} symbols={frame['symbol'].nunique() if not frame.empty else 0} "
          f"columns={len(frame.columns)}")
    if args.out and not frame.empty:
        frame.to_parquet(args.out) if args.out.endswith(".parquet") else frame.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_train(args) -> int:
    from dataclasses import asdict

    from stock_predictor.models import (
        ForwardModel, ModelCard, ModelRegistry, backend_name,
        model_version_string, training_set,
    )

    cfg = _target_cfg(args)
    with AnalyzerViews() as views:
        sessions = _sessions(views, args)
        if not sessions:
            print("no sessions to train on", file=sys.stderr)
            return 1
        step = args.minute_step
        print(f"target: {cfg.describe()}")
        print(f"assembling training set over {len(sessions)} sessions "
              f"({sessions[0]}..{sessions[-1]}), every {'minute' if step == 1 else f'{step} minutes'} ...")
        source = _frame_source(views, args, cfg=cfg, sessions=sessions)
        ts = training_set(views, sessions, minute_step=step, source=source,
                          progress=_progress("session"))
        print(_frames_line(source))
        if ts.rows == 0:
            print("empty training set", file=sys.stderr)
            return 1
        calibration, calibrator = [], None
        if cfg.fixed and args.calibrate > 0:
            calibrator, calibration = _calibrate(ts, source, cfg, args.calibrate)
    cols = ts.feature_names
    model = ForwardModel(feature_names=cols)
    on = get_params().active_features()
    print(f"candidate features on: {', '.join(on) if on else 'none'}")
    print(f"fitting ForwardModel: {ts.rows:,} rows x {len(cols)} features "
          f"({ts.X.nbytes / 1e6:,.0f} MB), backend={backend_name()}, "
          f"target hit on {(ts.label_class > 0).mean():.1%} of rows")
    model.fit_matrix(ts.X, ts.label_class, ts.forward_return_pct)
    model.calibrator = calibrator

    version = model_version_string(backend_name(), sessions[-1], cfg)
    card = ModelCard(
        model_version=version, backend=backend_name(), trained_through=sessions[-1],
        train_sessions=len(ts.sessions), feature_names=cols, params=get_params().to_dict(),
        minute_step=step, target=asdict(cfg), calibration=calibration,
    )
    path = ModelRegistry().save(version, model, card)
    print(f"saved {version} -> {path}")
    return 0


def _calibrate(ts, source, cfg, last_n: int):
    """Fit the classifier without the last ``last_n`` sessions, and map its
    P(target) on those sessions' prediction points to how often it came true.
    The model saved is then fitted on everything and carries this map."""
    import numpy as np

    from stock_predictor.eval.target import CALIBRATION_EDGES, default_points, point_rows
    from stock_predictor.models import GainCalibrator
    from stock_predictor.models.gbdt import GBDTClassifier

    if len(ts.sessions) < last_n + 20:
        print(f"too few sessions to calibrate on the last {last_n}; skipped")
        return None, []
    cut = len(ts.sessions) - last_n
    early = ts.row_session < cut
    print(f"calibrating: fitting on {ts.sessions[0]}..{ts.sessions[cut - 1]}, "
          f"checking {ts.sessions[cut]}..{ts.sessions[-1]} ...")
    clf = GBDTClassifier().fit(np.asfortranarray(ts.X[early]), (ts.label_class[early] > 0).astype(int))
    rows = point_rows(source, ts.sessions[cut:], default_points(cfg))
    p = np.asarray(clf.predict_proba(rows[ts.feature_names].to_numpy(dtype=np.float64)))
    hit = (rows["label_class"] > 0).to_numpy(dtype=float)
    cal = GainCalibrator().fit(p, hit)
    table = []
    for lo, hi in zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p <= hi)
        if m.any():
            table.append({"band": f"{lo:.2f}-{hi:.2f}", "rows": int(m.sum()),
                          "predicted": float(p[m].mean()), "seen": float(hit[m].mean())})
    for r in table:
        print(f"  P {r['band']}: predicted {r['predicted']:.1%}, came true {r['seen']:.1%} ({r['rows']} rows)")
    return cal, table


def cmd_evaluate(args) -> int:
    from stock_predictor.eval.backtest import backtest_baselines, backtest_picks
    from stock_predictor.eval.metrics import Pick

    params = get_params()
    with AnalyzerViews() as views:
        sessions = _sessions(views, args)
        ct = _clock_time(args.point, args.at)
        results = backtest_baselines(views, sessions, clock_time=ct, params=params)

        if args.model:
            from stock_predictor.models import ModelRegistry
            from stock_predictor.models.rank import rank_session

            version = args.model if args.model != "latest" else ModelRegistry().latest()
            if version:
                model, _card = ModelRegistry().load(version)
                source = _frame_source(views, args, params)

                def pick_fn(v, sd, ao):
                    top = rank_session(model, v, sd, ao, CONFIG.top_n, params,
                                       frame=source.at(sd, ao))
                    return [
                        Pick(symbol=s.symbol, gain_p25=s.gain_p25, gain_p50=s.gain_p50,
                             gain_p75=s.gain_p75, duration_bucket=s.duration_bucket)
                        for s in top
                    ]

                results[version] = backtest_picks(views, sessions, pick_fn, ct, params=params)
                print(_frames_line(source))

    filters = params.active_filters()
    if filters:
        print(f"filters on: {', '.join(filters)}")
    _print_table(results, _point_label(args), len(sessions), holdout=args.holdout)
    if args.record:
        _record(results, _point_label(args), args.holdout)
    return 0


def cmd_predict(args) -> int:
    import numpy as np

    from stock_predictor.features.pipeline import build_feature_frame, latest_row_per_symbol
    from stock_predictor.models import card_target

    version, model, card = _load_model(args.model)
    cfg = card_target(card)
    params = _card_params(card)
    with AnalyzerViews(live=True) as views:
        session = args.session or views.session_in_progress()
        day = date.fromisoformat(session)
        as_of = NSE.epoch_of(day, _clock_time("T1", args.at)) if args.at else views.last_finished_minute(session)
        if as_of is None:
            print(f"no candles for {session} yet", file=sys.stderr)
            return 1
        frame = build_feature_frame(views, session, as_of, params=params)
    if frame.empty:
        print(f"no features for {session} at {NSE.time_of(as_of):%H:%M}", file=sys.stderr)
        return 1
    rows = latest_row_per_symbol(frame)
    pred = model.predict(rows)
    rows = rows.assign(p=pred.p_up, p50=pred.gain_p50,
                       vol=rows["atr_14"] / rows["_close"].replace(0, np.nan) * 100)
    if pred.p_seen is not None:
        rows["seen"] = pred.p_seen
    rows["vol_rank"] = rows["vol"].rank(ascending=False, method="first").astype(int)
    rows = rows.sort_values("p", ascending=False).reset_index(drop=True)

    end = as_of + cfg.horizon_minutes * 60
    print(f"{version}: {cfg.describe()}")
    print(f"{session}, last finished candle {NSE.time_of(as_of):%H:%M} IST; "
          f"window to {NSE.time_of(end):%H:%M}")
    if end > NSE.session_close_epoch(day) - 60:
        print("  the window runs past the close: the model was never trained on such rows")
    head = f"{'':>3} {'symbol':<12} {'price':>10}"
    if cfg.fixed:
        head += f" {'target':>10}" + (f" {'stop':>10}" if cfg.stop_pct is not None else "")
    head += f" {'P(model)':>9}" + (f" {'P(seen)':>8}" if "seen" in rows else "")
    head += f" {'p50 %':>6} {'vol %':>6} {'vol#':>4}"
    print(head)
    for i, r in rows.head(args.n).iterrows():
        line = f"{i + 1:>3} {r['symbol']:<12} {r['_close']:>10.2f}"
        if cfg.fixed:
            line += f" {r['_close'] * (1 + cfg.target_pct / 100):>10.2f}"
            if cfg.stop_pct is not None:
                line += f" {r['_close'] * (1 - cfg.stop_pct / 100):>10.2f}"
        line += f" {r['p']:>9.1%}" + (f" {r['seen']:>8.1%}" if "seen" in rows else "")
        line += f" {r['p50']:>+6.2f} {r['vol']:>6.3f} {r['vol_rank']:>4}"
        print(line)
    print("P(model): the model's probability. P(seen): how often that probability came true on "
          "sessions it had not trained on.\nvol %: 14-minute ATR as % of price; vol#: its rank "
          "(most volatile first) -- the simple screen `evaluate-target` compares against.")
    if args.csv:
        rows.assign(as_of=as_of, model_version=version).to_csv(args.csv, index=False)
    return 0


def cmd_evaluate_target(args) -> int:
    from stock_predictor.eval.target import evaluate_target
    from stock_predictor.models import card_target

    version, model, card = _load_model(args.model)
    cfg = card_target(card)
    if not cfg.fixed:
        print(f"{version} was not trained on a fixed target (train --target)", file=sys.stderr)
        return 1
    with AnalyzerViews() as views:
        sessions = _sessions(views, args)
        seen = [s for s in sessions if s <= card.trained_through]
        if seen:
            print(f"warning: {len(seen)} of these sessions are on or before {card.trained_through}, "
                  "which the model was trained through: in-sample")
        source = _frame_source(views, args, _card_params(card), cfg, sessions)
        rep = evaluate_target(model, source, sessions, n=args.n)
        print(_frames_line(source))

    import pandas as pd

    t = rep.by_point
    print(f"\n{version}: {cfg.describe()}; top {rep.n} at each point over {len(rep.sessions)} sessions "
          f"({rep.sessions[0]}..{rep.sessions[-1]}); costs {rep.cost_pct:.3f}%")
    cols = ["point", "base_hit", "model_hit", "intraday_volatility_hit", "highest_volatility_hit",
            "model_p", "model_net", "intraday_volatility_net", "highest_volatility_net"]
    if cfg.stop_pct is not None:
        cols.insert(5, "model_stopped")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\ncalibration, every symbol at every point:")
        print(rep.calibration.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n" + json.dumps(rep.summary, indent=1))
    if args.csv:
        t.to_csv(args.csv, index=False)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep.to_dict(), fh, indent=1)
    return 0


def cmd_backtest(args) -> int:
    from stock_predictor.eval.backtest import backtest_baselines
    from stock_predictor.eval.splits import walk_forward_folds

    with AnalyzerViews() as views:
        all_sessions = _sessions(views, args)
        folds = walk_forward_folds(all_sessions, initial_train=20, test_size=5)
        ct = _clock_time(args.point, args.at)
        print(f"{len(folds)} walk-forward folds over {len(all_sessions)} sessions")
        agg: dict[str, list] = {}
        for fold in folds:
            res = backtest_baselines(views, fold.test_sessions, clock_time=ct, params=get_params())
            for name, m in res.items():
                agg.setdefault(name, []).append(m)
        from stock_predictor.eval.metrics import aggregate as agg_metrics

        summary = {name: agg_metrics(ms) for name, ms in agg.items()}
    _print_table(summary, _point_label(args), len(all_sessions), holdout=False)
    return 0


def cmd_ablate(args) -> int:
    from stock_predictor.eval.ablation import format_report, run_ablation

    params = get_params()
    model = None
    score_label = "20-day ATR %"
    if args.model:
        from stock_predictor.models import ModelRegistry

        version = args.model if args.model != "latest" else ModelRegistry().latest()
        if not version:
            print("no trained model found", file=sys.stderr)
            return 1
        model, _card = ModelRegistry().load(version)
        score_label = f"model score ({version})"

    with AnalyzerViews() as views:
        sessions = _sessions(views, args)
        ct = _clock_time("T1", args.at)
        source = score_fn = None
        if model is not None:
            from stock_predictor.models.rank import rank_session

            source = _frame_source(views, args, params)

            def score_fn(v, sd, ao):
                top = rank_session(model, v, sd, ao, 10**6, params, frame=source.at(sd, ao))
                return {p.symbol: p.score for p in top}

        report = run_ablation(views, sessions, clock_time=ct, params=params, model=model,
                              score_fn=score_fn, score_label=score_label,
                              progress=_progress("session"), frames=source)
    if source is not None:
        print(_frames_line(source), file=sys.stderr)
    print(format_report(report))
    if args.json:
        import dataclasses

        with open(args.json, "w") as fh:
            json.dump(dataclasses.asdict(report), fh, indent=2, default=float)
        print(f"\nwrote {args.json}")
    return 0


def cmd_params(args) -> int:
    p = get_params()
    if args.json:
        print(json.dumps(p.to_dict(), indent=2))
        return 0
    for section, values in p.to_dict().items():
        print(f"[{section}]")
        for k, v in values.items():
            print(f"{k} = {_toml_value(v)}")
        print()
    return 0


def cmd_frames(args) -> int:
    from stock_predictor.data.framestore import FrameStore

    store = FrameStore()
    usage = store.usage()
    if not usage:
        print(f"no saved frames under {store.root}")
        return 0
    total = 0
    for variant, u in usage.items():
        total += u["bytes"]
        desc = json.loads((store.root / variant / "variant.json").read_text())
        on = ", ".join(desc.get("features", {})) or "no candidate features"
        print(f"{variant}  {u['files']:>4} sessions  {u['bytes'] / 1e6:>8.1f} MB  ({on})")
    print(f"total {total / 1e6:.1f} MB under {store.root}")
    return 0


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(v)


def cmd_serve(args) -> int:
    import uvicorn

    from stock_predictor.serving.app import create_app
    from stock_predictor.serving.scheduler import Scheduler
    from stock_predictor.serving.service import PredictorService

    # live: today's candles come from v_intraday_merged while the session runs
    svc = PredictorService(views=AnalyzerViews(live=True))
    if args.model:
        from stock_predictor.models import ModelRegistry

        reg = ModelRegistry()
        version = args.model if args.model != "latest" else reg.latest()
        if version:
            svc.model, _ = reg.load(version)
            svc.model_version = version

    sched = Scheduler(svc.recompute)
    today = args.session or date.today().isoformat()
    try:
        svc.recompute(today, NSE.epoch_of(date.fromisoformat(today), OPENING_RANGE_T1))
    except Exception as exc:  # pragma: no cover
        print(f"warm recompute skipped: {exc}", file=sys.stderr)
    sched.run_forever(session_date=args.session)
    app = create_app(svc)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    sched.stop()
    return 0


def cmd_score(args) -> int:
    from stock_predictor.serving.service import PredictorService

    svc = PredictorService()
    today = args.today or date.today().isoformat()
    n = svc.score_yesterday(today)
    print(f"scored {n} pending prediction rows against realised outcomes")
    return 0


# -- output helpers ---------------------------------------------------------


_METRIC_ORDER = ["realised_net_gain", "precision_at_k", "ndcg_at_k", "quantile_coverage",
                 "duration_accuracy", "adverse_rate", "hits", "sessions_scored"]


def _print_table(results: dict, point: str, n_sessions: int, holdout: bool) -> None:
    tag = "HELD-OUT" if holdout else "development"
    print(f"\nprediction point {point} | {n_sessions} {tag} sessions\n")
    head = f"{'strategy':<26}" + "".join(f"{m[:14]:>16}" for m in _METRIC_ORDER)
    print(head)
    print("-" * len(head))
    ranked = sorted(results.items(), key=lambda kv: kv[1].get("realised_net_gain", 0), reverse=True)
    for name, m in ranked:
        row = f"{name:<26}" + "".join(f"{m.get(k, 0):>16.4f}" for k in _METRIC_ORDER)
        print(row)
    print()


def _record(results: dict, point: str, holdout: bool) -> None:
    import time

    fold = "holdout" if holdout else f"dev@{point}"
    with Store() as store:
        now = int(time.time())
        for name, m in results.items():
            kind = "baseline" if "triple-barrier" not in name else "model"
            for metric, value in m.items():
                store.record_metric(now, kind, name, fold, metric, float(value))
    print("recorded metrics to the store")


# -- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stock-predictor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", metavar="FILE",
                   help="parameters file (.toml or .json); default $STOCK_PREDICTOR_PARAMS")
    p.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                   help="override one parameter, e.g. --set rsi.filter=true (repeatable)")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", help="first session (YYYY-MM-DD)")
    common.add_argument("--end", help="last session (YYYY-MM-DD)")
    common.add_argument("--holdout", action="store_true", help="use the held-out sessions")
    common.add_argument("--max-sessions", type=int, help="cap to the last N sessions")
    common.add_argument("--no-frame-cache", action="store_true",
                        help="rebuild every session frame and save none")
    at_help = "prediction time HH:MM (overrides --point), e.g. 09:45"

    bf = sub.add_parser("build-features", help="build one session's feature frame")
    bf.add_argument("session")
    bf.add_argument("--point", default="T1", choices=["T0", "T1", "Tn"])
    bf.add_argument("--out", help="write the frame to .csv or .parquet")
    bf.set_defaults(func=cmd_build_features)

    from stock_predictor.models.training import default_jobs

    jobs_help = f"processes for building missing frames (default {default_jobs()})"

    tr = sub.add_parser("train", parents=[common], help="fit the stage-1 model")
    tr.add_argument("--minute-step", type=int, default=1, metavar="K",
                    help="train on every K-th minute after the open (default 1: every minute)")
    tr.add_argument("--horizon", type=int, metavar="MIN",
                    help="label window in minutes (default 30)")
    tr.add_argument("--target", type=float, metavar="PCT",
                    help="fixed target: label = price touches +PCT%% within the horizon "
                         "(default: ATR-scaled barriers)")
    tr.add_argument("--stop", type=float, metavar="PCT",
                    help="with --target: the target must come before -PCT%%")
    tr.add_argument("--calibrate", type=int, default=20, metavar="N",
                    help="with --target: map P to the rate seen on the last N sessions, "
                         "fitted without them (default 20; 0 = off)")
    tr.add_argument("--all-sessions", action="store_true",
                    help="train on the held-out sessions too (for the model you use live, "
                         "after evaluating one trained without them)")
    tr.add_argument("--jobs", type=int, default=default_jobs(), help=jobs_help)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="score the session in progress with a trained model")
    pr.add_argument("--model", default="latest", help="'latest' or a version")
    pr.add_argument("--session", help="a past session instead of today's (YYYY-MM-DD)")
    pr.add_argument("--at", help="HH:MM (default: the latest finished candle)")
    pr.add_argument("-n", type=int, default=15, help="rows to show (default 15)")
    pr.add_argument("--csv", help="also write every symbol's row")
    pr.set_defaults(func=cmd_predict)

    et = sub.add_parser("evaluate-target", parents=[common],
                        help="how often a fixed-target model's top picks reach the target")
    et.add_argument("--model", default="latest", help="'latest' or a version")
    et.add_argument("-n", type=int, default=5, help="picks per point (default 5)")
    et.add_argument("--jobs", type=int, default=default_jobs(), help=jobs_help)
    et.add_argument("--csv", help="write the per-point table")
    et.add_argument("--json", help="write the summary")
    et.set_defaults(func=cmd_evaluate_target)

    ev = sub.add_parser("evaluate", parents=[common], help="baselines + model at a prediction point")
    ev.add_argument("--point", default="T1", choices=["T0", "T1", "Tn"])
    ev.add_argument("--at", help=at_help)
    ev.add_argument("--model", nargs="?", const="latest", help="also score a model ('latest' or a version)")
    ev.add_argument("--record", action="store_true", help="write metrics to the store")
    ev.set_defaults(func=cmd_evaluate)

    bt = sub.add_parser("backtest", parents=[common], help="walk-forward folds")
    bt.add_argument("--point", default="T1", choices=["T0", "T1", "Tn"])
    bt.add_argument("--at", help=at_help)
    bt.set_defaults(func=cmd_backtest)

    ab = sub.add_parser("ablate", parents=[common],
                        help="measure each candidate screening input on this project's target")
    ab.add_argument("--at", default="09:45",
                    help="prediction time HH:MM (default 09:45, so the opening check is known)")
    ab.add_argument("--model", nargs="?", const="latest",
                    help="use a trained model's score for the score-order quintiles")
    ab.add_argument("--json", help="also write the report as JSON")
    ab.set_defaults(func=cmd_ablate)

    pa = sub.add_parser("params", help="print the effective parameters (TOML)")
    pa.add_argument("--json", action="store_true", help="print JSON instead")
    pa.set_defaults(func=cmd_params)

    fr = sub.add_parser("frames", help="show the saved training frames and their disk use")
    fr.set_defaults(func=cmd_frames)

    sv = sub.add_parser("serve", help="run the REST service")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8001)
    sv.add_argument("--session", help="pin the session date (default: today)")
    sv.add_argument("--model", nargs="?", const="latest", help="serve a trained model")
    sv.set_defaults(func=cmd_serve)

    sc = sub.add_parser("score", help="fill realised gains on past predictions")
    sc.add_argument("--today", help="treat this date as 'now' (default: today)")
    sc.set_defaults(func=cmd_score)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        params = load_params(args.params)
        if args.set:
            params = params.with_overrides(parse_set(args.set))
    except (ValueError, OSError) as exc:
        parser.error(f"params: {exc}")
    set_params(params)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
