"""``stock-predictor`` command line.

    build-features   build the feature frame for one session / prediction point
    train            fit the stage-1 model on the development sessions
    evaluate         baselines (and the model, if trained) over a prediction point
    backtest         evaluate across purged walk-forward folds
    serve            run the REST service + recompute scheduler
    score            fill realised gains on past predictions
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


def _clock_time(name: str):
    from stock_predictor.config import LAST_INTRADAY_POINT

    return {"T0": PREOPEN_T0, "T1": OPENING_RANGE_T1, "Tn": LAST_INTRADAY_POINT}.get(name, OPENING_RANGE_T1)


def _sessions(views: AnalyzerViews, args) -> list[str]:
    s = views.sessions(start=getattr(args, "start", None), end=getattr(args, "end", None))
    from stock_predictor.eval.splits import holdout_split

    dev, hold = holdout_split(s, CONFIG.holdout_sessions)
    chosen = hold if getattr(args, "holdout", False) else dev
    if getattr(args, "max_sessions", None):
        chosen = chosen[-args.max_sessions :]
    return chosen


# -- commands -----------------------------------------------------------------


def cmd_build_features(args) -> int:
    from stock_predictor.features.pipeline import build_feature_frame

    with AnalyzerViews() as views:
        as_of = NSE.epoch_of(date.fromisoformat(args.session), _clock_time(args.point))
        frame = build_feature_frame(views, args.session, as_of)
    print(f"session={args.session} point={args.point} as_of={as_of}")
    print(f"rows={len(frame)} symbols={frame['symbol'].nunique() if not frame.empty else 0} "
          f"columns={len(frame.columns)}")
    if args.out and not frame.empty:
        frame.to_parquet(args.out) if args.out.endswith(".parquet") else frame.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_train(args) -> int:
    from stock_predictor.models import (
        ForwardModel, ModelCard, ModelRegistry, backend_name,
        model_feature_columns, model_version_string, training_frame,
    )

    with AnalyzerViews() as views:
        sessions = _sessions(views, args)
        if not sessions:
            print("no sessions to train on", file=sys.stderr)
            return 1
        print(f"assembling training frame over {len(sessions)} sessions "
              f"({sessions[0]}..{sessions[-1]}) ...")
        frame = training_frame(views, sessions)
    if frame.empty:
        print("empty training frame", file=sys.stderr)
        return 1
    cols = model_feature_columns(frame)
    model = ForwardModel(feature_names=cols)
    print(f"fitting ForwardModel: {len(frame):,} rows x {len(cols)} features, backend={backend_name()}")
    model.fit(frame, frame["label_class"], frame["forward_return_pct"])

    version = model_version_string(backend_name(), sessions[-1])
    card = ModelCard(
        model_version=version, backend=backend_name(), trained_through=sessions[-1],
        train_sessions=len(sessions), feature_names=cols,
    )
    path = ModelRegistry().save(version, model, card)
    print(f"saved {version} -> {path}")
    return 0


def cmd_evaluate(args) -> int:
    from stock_predictor.eval.backtest import backtest_baselines, backtest_picks
    from stock_predictor.eval.metrics import Pick

    with AnalyzerViews() as views:
        sessions = _sessions(views, args)
        ct = _clock_time(args.point)
        results = backtest_baselines(views, sessions, clock_time=ct)

        if args.model:
            from stock_predictor.models import ModelRegistry
            from stock_predictor.features.pipeline import build_feature_frame
            from stock_predictor.models.aggregate import PerMinutePrediction, aggregate_to_topn

            version = args.model if args.model != "latest" else ModelRegistry().latest()
            if version:
                model, _card = ModelRegistry().load(version)

                def pick_fn(v, sd, ao):
                    f = build_feature_frame(v, sd, ao)
                    if f.empty:
                        return []
                    pr = model.predict(f)
                    pm = [
                        PerMinutePrediction(
                            symbol=f.iloc[i]["symbol"], ts_epoch=int(f.iloc[i]["ts_epoch"]),
                            p_up=float(pr.p_up[i]), gain_p25=float(pr.gain_p25[i]),
                            gain_p50=float(pr.gain_p50[i]), gain_p75=float(pr.gain_p75[i]),
                        )
                        for i in range(len(f))
                    ]
                    top = aggregate_to_topn(pm, CONFIG.top_n)
                    return [
                        Pick(symbol=s.symbol, gain_p25=s.gain_p25, gain_p50=s.gain_p50,
                             gain_p75=s.gain_p75, duration_bucket=s.duration_bucket)
                        for s in top
                    ]

                results[version] = backtest_picks(views, sessions, pick_fn, ct)

    _print_table(results, args.point, len(sessions), holdout=args.holdout)
    if args.record:
        _record(results, args.point, args.holdout)
    return 0


def cmd_backtest(args) -> int:
    from stock_predictor.eval.backtest import backtest_baselines
    from stock_predictor.eval.splits import walk_forward_folds

    with AnalyzerViews() as views:
        all_sessions = _sessions(views, args)
        folds = walk_forward_folds(all_sessions, initial_train=20, test_size=5)
        ct = _clock_time(args.point)
        print(f"{len(folds)} walk-forward folds over {len(all_sessions)} sessions")
        agg: dict[str, list] = {}
        for fold in folds:
            res = backtest_baselines(views, fold.test_sessions, clock_time=ct)
            for name, m in res.items():
                agg.setdefault(name, []).append(m)
        from stock_predictor.eval.metrics import aggregate as agg_metrics

        summary = {name: agg_metrics(ms) for name, ms in agg.items()}
    _print_table(summary, args.point, len(all_sessions), holdout=False)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from stock_predictor.serving.app import create_app
    from stock_predictor.serving.scheduler import Scheduler
    from stock_predictor.serving.service import PredictorService

    svc = PredictorService()
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
                 "duration_accuracy", "hits", "sessions_scored"]


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
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", help="first session (YYYY-MM-DD)")
    common.add_argument("--end", help="last session (YYYY-MM-DD)")
    common.add_argument("--holdout", action="store_true", help="use the held-out sessions")
    common.add_argument("--max-sessions", type=int, help="cap to the last N sessions")

    bf = sub.add_parser("build-features", help="build one session's feature frame")
    bf.add_argument("session")
    bf.add_argument("--point", default="T1", choices=["T0", "T1", "Tn"])
    bf.add_argument("--out", help="write the frame to .csv or .parquet")
    bf.set_defaults(func=cmd_build_features)

    tr = sub.add_parser("train", parents=[common], help="fit the stage-1 model")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", parents=[common], help="baselines + model at a prediction point")
    ev.add_argument("--point", default="T1", choices=["T0", "T1", "Tn"])
    ev.add_argument("--model", nargs="?", const="latest", help="also score a model ('latest' or a version)")
    ev.add_argument("--record", action="store_true", help="write metrics to the store")
    ev.set_defaults(func=cmd_evaluate)

    bt = sub.add_parser("backtest", parents=[common], help="walk-forward folds")
    bt.add_argument("--point", default="T1", choices=["T0", "T1", "Tn"])
    bt.set_defaults(func=cmd_backtest)

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
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
