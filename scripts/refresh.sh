#!/usr/bin/env bash
# refresh.sh: the daily refresh, once stock-analyzer has stored the day's candles.
#
#   1. waits (up to --wait minutes, default 120) until stock-analyzer's evening batch has
#      stored today's session; a day with no session (weekend, holiday) skips straight on
#   2. scores the picks today.sh and tomorrow.sh recorded about today (stock-predictor scorecard)
#   3. retrains the models today.sh and tomorrow.sh use on every finished session, and the
#      last-month ones if you use them; the new session's saved frame is built on the way
#   4. on Fridays, or with --evaluate, redoes the held-out checks behind "how much to trust it"
#   5. deletes superseded models, keeping the newest 3 of each kind (--keep N)
#
# Every step is idempotent: running it twice in an evening does the work once.
# scripts/schedule-refresh.sh install runs it on weekdays at 19:15, after stock-analyzer's
# 18:30 batch. Output goes to data/logs/refresh.log as well as the terminal.
#
#   scripts/refresh.sh                 the refresh
#   scripts/refresh.sh --evaluate      also redo the held-out checks today
#   scripts/refresh.sh --target 1      for a different gain than 0.5%
#   scripts/refresh.sh --wait 0        do not wait for stock-analyzer: refresh with what is stored
set -eo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
sp="$root/.venv/bin/stock-predictor"
py="$root/.venv/bin/python"
[ -x "$sp" ] || { echo "refresh.sh: $root/.venv not found" >&2; exit 1; }

target=0.5; wait_minutes=120; evaluate=0; keep=3
while [ $# -gt 0 ]; do
  case "$1" in
    --target)   target="$2"; shift 2 ;;
    --wait)     wait_minutes="$2"; shift 2 ;;
    --evaluate) evaluate=1; shift ;;
    --keep)     keep="$2"; shift 2 ;;
    -h|--help)  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "refresh.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[ "$(date +%u)" = 5 ] && evaluate=1

mkdir -p "$root/data/logs"
exec > >(tee -a "$root/data/logs/refresh.log") 2>&1
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }

# 1. Has stock-analyzer stored today's session?
state() {
  "$py" - <<'PY'
from stock_predictor.data.analyzer import AnalyzerViews
v = AnalyzerViews(live=True)
today = v.session_in_progress()
finished = v.sessions()
if finished and finished[-1] == today:
    print("stored", today)
elif v.last_finished_minute(today) is not None:
    print("waiting", today)
else:
    print("none", today)
PY
}
say "refresh: target +$target%"
read -r st today <<<"$(state)"
waited=0
while [ "$st" = waiting ] && [ "$waited" -lt "$wait_minutes" ]; do
  say "stock-analyzer has not stored $today yet; checking again in 5 minutes"
  sleep 300; waited=$((waited + 5))
  read -r st today <<<"$(state)"
done
case "$st" in
  stored)  say "stock-analyzer has stored $today" ;;
  none)    say "no session today ($today): nothing new to learn from" ;;
  waiting)
    if [ "$wait_minutes" = 0 ]; then
      say "$today is not stored yet; refreshing with the sessions stored so far"
    else
      say "gave up after $wait_minutes minutes: $today is still not stored; run this again later"; exit 75
    fi ;;
esac

# 2. Score what was predicted about today, and anything else closed but unscored.
if [ "$st" = stored ]; then
  "$sp" scorecard --session "$today" || say "scorecard failed (continuing)"
fi
"$sp" scorecard --days 31 >/dev/null || true

# 3. Bring every model in use up to date (each is a no-op if it already is).
say "today.sh's model"
scripts/today.sh --target "$target" --train-only --no-record
say "tomorrow.sh's model"
scripts/today.sh --target "$target" --tomorrow --train-only --no-record
for flags in "--recent" "--recent --tomorrow"; do
  # the last-month models only if one was ever trained
  in_use="$("$py" -c "from stock_predictor.serving.summary import find_model; from stock_predictor.models import ModelRegistry; print(bool(find_model(ModelRegistry(), $target, True, '--tomorrow' in '$flags')))")"
  if [ "$in_use" = True ]; then
    say "last-month model ($flags)"
    # shellcheck disable=SC2086
    scripts/today.sh --target "$target" $flags --train-only --no-record
  fi
done

# 4. The held-out checks: the last 21 sessions move on every day.
if [ "$evaluate" = 1 ]; then
  say "held-out checks"
  scripts/today.sh --target "$target" --evaluate --train-only --no-record
  scripts/today.sh --target "$target" --tomorrow --evaluate --train-only --no-record
fi

# 5. Superseded models.
"$sp" models --prune "$keep"
say "refresh done"
