#!/usr/bin/env bash
# schedule-today.sh: run scripts/today.sh every 10 minutes from 09:20 to 14:50 on weekdays, with
# launchd (macOS). Each run records its picks (so the scorecard and the Grafana dashboards fill up)
# and appends its summary to data/logs/today/<date>.log.
#
#   scripts/schedule-today.sh install                 write ~/Library/LaunchAgents/com.stockpredictor.today.plist, load it
#   scripts/schedule-today.sh install --stop 0.5      the same, passing these options to every today.sh run
#   scripts/schedule-today.sh uninstall               unload and remove it
#   scripts/schedule-today.sh status                  whether it is loaded, and the end of today's log
#   scripts/schedule-today.sh plist [options]         print what install would write, and change nothing
#   scripts/schedule-today.sh run [today.sh options]  one run as the schedule does it (what launchd calls)
#
# A run is skipped on an NSE holiday (stock-analyzer's list), and while the previous one is still
# going (the first run after a missed evening refresh retrains, about 3 minutes). Runs missed while
# the machine was asleep start as one when it wakes. Times are wall clock: keep the machine on IST.
set -eo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
label=com.stockpredictor.today
plist="$HOME/Library/LaunchAgents/$label.plist"
logs="$root/data/logs/today"
first=$((9 * 60 + 20)); last=$((14 * 60 + 50)); step=10

xml_escape() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

write_plist() {
  local slots="" d m args=""
  for d in 1 2 3 4 5; do
    for ((m = first; m <= last; m += step)); do
      slots="$slots    <dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$((m / 60))</integer><key>Minute</key><integer>$((m % 60))</integer></dict>
"
    done
  done
  for a in "$@"; do args="$args    <string>$(xml_escape "$a")</string>
"; done
  mkdir -p "$(dirname "$plist")" "$logs"
  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$root/scripts/schedule-today.sh</string>
    <string>run</string>
$args  </array>
  <key>WorkingDirectory</key>
  <string>$root</string>
  <key>StartCalendarInterval</key>
  <array>
$slots  </array>
  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>$root/data/logs/today.launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>$root/data/logs/today.launchd.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin</string>
  </dict>
</dict>
</plist>
PLIST
}

is_trading_day() {  # prints "yes", "weekend" or "holiday" for today (IST)
  "$root/.venv/bin/python" - <<'PY'
from datetime import date
from stock_predictor.clock import NSE
from stock_predictor.models.nextday import holidays
import time
d = NSE.session_date_of(int(time.time()))
hs = holidays(d.year)
print("weekend" if d.weekday() >= 5 else "holiday" if hs and d in hs else "yes")
PY
}

case "$1" in
  install)
    shift
    write_plist "$@"
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load -w "$plist"
    printf 'installed %s: weekdays every %d minutes, %02d:%02d to %02d:%02d%s\n' "$label" "$step" \
      $((first / 60)) $((first % 60)) $((last / 60)) $((last % 60)) "${*:+, with: $*}"
    ;;
  plist)
    shift
    plist=/dev/stdout write_plist "$@"
    ;;
  uninstall)
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "removed $label"
    ;;
  status)
    if launchctl list | grep -q "$label"; then echo "$label is loaded ($plist)"; else echo "$label is not loaded"; fi
    today_log="$logs/$(date +%F).log"
    if [ -f "$today_log" ]; then
      echo "--- runs today: $(grep -c '^=== ' "$today_log")"
      grep '^=== ' "$today_log" | tail -n 3
    else
      echo "--- no run logged today ($today_log)"
    fi
    ;;
  run)
    shift
    mkdir -p "$logs"
    log="$logs/$(date +%F).log"
    day="$(is_trading_day)"
    if [ "$day" != yes ]; then
      echo "=== $(date +%H:%M) skipped: $day" >> "$log"
      exit 0
    fi
    lock="$root/data/logs/.today.lock"
    if ! mkdir "$lock" 2>/dev/null; then
      if [ -n "$(find "$lock" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then
        rm -rf "$lock"; mkdir "$lock"  # left behind by a run that died
      else
        echo "=== $(date +%H:%M) skipped: the previous run is still going" >> "$log"
        exit 0
      fi
    fi
    trap 'rm -rf "$lock"' EXIT
    {
      echo "=== $(date +%H:%M)"
      "$root/scripts/today.sh" "$@" 2>&1 || echo "(today.sh exited with $?)"
      echo
    } >> "$log"
    ;;
  *)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    [ -z "$1" ] || [ "$1" = -h ] || [ "$1" = --help ] || exit 2
    ;;
esac
