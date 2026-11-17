#!/usr/bin/env bash
# today.sh: which stocks are expected to gain +0.5% or more before today's
# close, by how much, and how far to trust it. Run it whenever you want it.
#
#   scripts/today.sh                   the summary, from the latest finished candle
#   scripts/today.sh --target 1        a different gain (trains a model for it if needed)
#   scripts/today.sh --min-chance 60   a stricter filter (default 50)
#   scripts/today.sh --stop 0.5        +0.5% before it falls 0.5% (by the close), from a model trained on
#                                      that question (trains one if needed; today's session only)
#   scripts/today.sh --min-edge 15     flag stocks whose chance is 15+ points above what any stock gets
#                                      at this time of day (default 10)
#   scripts/today.sh --top 20          list more stocks in full (default 15)
#   scripts/today.sh --at 11:00        as of an earlier minute today
#   scripts/today.sh --session 2026-11-09 --at 10:30    a past day
#   scripts/today.sh --json out.json   also write the summary as JSON
#   scripts/today.sh --retrain         retrain on every finished session first (~2 min)
#   scripts/today.sh --no-retrain      never retrain, even if a session has finished since
#   scripts/today.sh --evaluate        redo the held-out check behind "how much to trust it" (~3 min)
#   scripts/today.sh --tomorrow        the next session instead: +0.5% from its open, judged from each
#                                      stock's state at 15:00 today (scripts/tomorrow.sh does the same)
#   scripts/today.sh --train-only      bring the model up to date and stop (scripts/refresh.sh uses it)
#   scripts/today.sh --no-record       do not store this run's predictions (stored by default, so
#                                      `stock-predictor scorecard` and tomorrow.sh can score them)
#
# The model is retrained by itself when a session has finished since it was
# trained. Needs this repo's .venv and stock-analyzer's database beside it.
set -eo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
sp="$root/.venv/bin/stock-predictor"
py="$root/.venv/bin/python"
if [ ! -x "$sp" ] || [ ! -x "$py" ]; then
  echo "today.sh: $root/.venv not found; create it and run: pip install -e '.[all]'" >&2
  exit 1
fi

target=0.5
stop=""
tomorrow=0
train_only=0
record=1
retrain=auto
evaluate=0
pass=()
while [ $# -gt 0 ]; do
  case "$1" in
    --target)     target="$2"; pass+=(--target "$2"); shift 2 ;;
    --stop)       stop="$2"; pass+=(--stop "$2"); shift 2 ;;
    --retrain)    retrain=yes; shift ;;
    --no-retrain) retrain=no; shift ;;
    --evaluate)   evaluate=1; shift ;;
    --tomorrow)   tomorrow=1; shift ;;
    --train-only) train_only=1; shift ;;
    --no-record)  record=0; shift ;;
    -h|--help)    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *)            pass+=("$1"); shift ;;
  esac
done
case "$target" in ''|*[!0-9.]*) echo "today.sh: --target must be a number, e.g. 0.5" >&2; exit 2 ;; esac
case "$stop" in *[!0-9.]*) echo "today.sh: --stop must be a number, e.g. 0.5" >&2; exit 2 ;; esac
if [ -n "$stop" ] && [ "$tomorrow" = 1 ]; then
  echo "today.sh: --stop is for today's session only, not --tomorrow" >&2; exit 2
fi

if [ "$tomorrow" = 1 ]; then
  # One row per stock and session: the state at 15:00 against the session after.
  train=(train --target "$target" --next-session)
  pass+=(--next-session)
  is_next=True
else
  # Every finished session stock-analyzer has, every minute of each.
  train=(train --target "$target" --to-close --band max-gain --minute-step 1)
  [ -n "$stop" ] && train+=(--stop "$stop")
  is_next=False
fi
stop_py="${stop:-None}"
what="+$target% before the close${stop:+, before -$stop%}"
say() { printf '%s\n' "$*" >&2; }

if [ "$evaluate" = 1 ]; then
  say "Held-out check: training on all but the last 21 sessions, then testing on them ..."
  log="$(mktemp)"
  "$sp" "${train[@]}" >"$log" 2>&1 || { cat "$log" >&2; exit 1; }
  held="$(sed -n 's/^saved \([^ ]*\) .*/\1/p' "$log")"
  rm -f "$log"
  "$sp" evaluate-target --holdout --model "$held" --record >/dev/null
  say "  recorded beside $held"
fi

state="$("$py" -c "from stock_predictor.serving.summary import model_state; print(model_state($target, next_session=$is_next, stop_pct=$stop_py))")"
if [ "$retrain" = yes ] || { [ "$retrain" = auto ] && [ "$state" != ok ]; }; then
  case "$state" in
    missing) say "No model for $what yet: training one on every finished session ..." ;;
    stale)   say "A session has finished since the model was trained: retraining ..." ;;
    *)       say "Retraining on every finished session ..." ;;
  esac
  log="$(mktemp)"
  "$sp" "${train[@]}" --all-sessions >"$log" 2>&1 || { cat "$log" >&2; exit 1; }
  sed -n 's/^saved \([^ ]*\) .*/  saved \1/p' "$log" >&2
  rm -f "$log"
  say ""
fi

if [ "$train_only" = 1 ]; then
  [ "$state" = ok ] && [ "$retrain" != yes ] && say "Up to date: the model was trained through the latest finished session."
  exit 0
fi
[ "$record" = 1 ] && pass+=(--record)
if [ "$tomorrow" = 1 ]; then
  # First, how the picks recorded for today did (or are doing).
  "$sp" scorecard || true
  echo
fi
exec "$sp" summary ${pass[@]+"${pass[@]}"}
