#!/usr/bin/env bash
# today.sh: which stocks are expected to gain +0.5% or more before today's
# close, by how much, and how far to trust it. Run it whenever you want it.
#
#   scripts/today.sh                   the summary, from the latest finished candle
#   scripts/today.sh --target 1        a different gain (trains a model for it if needed)
#   scripts/today.sh --min-chance 60   a stricter filter (default 50)
#   scripts/today.sh --top 20          list more stocks in full (default 15)
#   scripts/today.sh --at 11:00        as of an earlier minute today
#   scripts/today.sh --session 2026-11-09 --at 10:30    a past day
#   scripts/today.sh --json out.json   also write the summary as JSON
#   scripts/today.sh --retrain         retrain on every finished session first (~2 min)
#   scripts/today.sh --no-retrain      never retrain, even if a session has finished since
#   scripts/today.sh --evaluate        redo the held-out check behind "how much to trust it" (~3 min)
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
retrain=auto
evaluate=0
pass=()
while [ $# -gt 0 ]; do
  case "$1" in
    --target)     target="$2"; pass+=(--target "$2"); shift 2 ;;
    --retrain)    retrain=yes; shift ;;
    --no-retrain) retrain=no; shift ;;
    --evaluate)   evaluate=1; shift ;;
    -h|--help)    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            pass+=("$1"); shift ;;
  esac
done
case "$target" in ''|*[!0-9.]*) echo "today.sh: --target must be a number, e.g. 0.5" >&2; exit 2 ;; esac

train=(train --target "$target" --to-close --band max-gain --minute-step 3)
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

state="$("$py" -c "from stock_predictor.serving.summary import model_state; print(model_state($target))")"
if [ "$retrain" = yes ] || { [ "$retrain" = auto ] && [ "$state" != ok ]; }; then
  case "$state" in
    missing) say "No model for +$target% before the close yet: training one on every finished session ..." ;;
    stale)   say "A session has finished since the model was trained: retraining ..." ;;
    *)       say "Retraining on every finished session ..." ;;
  esac
  log="$(mktemp)"
  "$sp" "${train[@]}" --all-sessions >"$log" 2>&1 || { cat "$log" >&2; exit 1; }
  sed -n 's/^saved \([^ ]*\) .*/  saved \1/p' "$log" >&2
  rm -f "$log"
  say ""
fi

exec "$sp" summary ${pass[@]+"${pass[@]}"}
