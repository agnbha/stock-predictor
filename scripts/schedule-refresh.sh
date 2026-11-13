#!/usr/bin/env bash
# schedule-refresh.sh: run scripts/refresh.sh every weekday at 19:15, with launchd (macOS),
# 45 minutes after stock-analyzer's own 18:30 evening batch (refresh.sh waits if it is late).
#
#   scripts/schedule-refresh.sh install     write ~/Library/LaunchAgents/com.stockpredictor.refresh.plist and load it
#   scripts/schedule-refresh.sh uninstall   unload and remove it
#   scripts/schedule-refresh.sh status      whether it is loaded, and the end of the last refresh log
#   scripts/schedule-refresh.sh --at HH:MM install    another time (wall clock: keep the machine on IST)
#
# A run missed while the machine was asleep starts when it wakes.
set -eo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
label=com.stockpredictor.refresh
plist="$HOME/Library/LaunchAgents/$label.plist"
hour=19; minute=15

if [ "$1" = --at ]; then
  case "$2" in [0-2][0-9]:[0-5][0-9]) hour=$((10#${2%%:*})); minute=$((10#${2##*:})) ;;
    *) echo "schedule-refresh.sh: --at takes HH:MM" >&2; exit 2 ;; esac
  shift 2
fi

write_plist() {
  local days=""
  for d in 1 2 3 4 5; do
    days="$days    <dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
"
  done
  mkdir -p "$(dirname "$plist")" "$root/data/logs"
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
    <string>$root/scripts/refresh.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$root</string>
  <key>StartCalendarInterval</key>
  <array>
$days  </array>
  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>$root/data/logs/refresh.launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>$root/data/logs/refresh.launchd.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin</string>
  </dict>
</dict>
</plist>
PLIST
}

case "$1" in
  install)
    write_plist
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load -w "$plist"
    printf 'installed %s: weekdays at %02d:%02d\n' "$label" "$hour" "$minute"
    ;;
  uninstall)
    launchctl unload "$plist" 2>/dev/null || true
    rm -f "$plist"
    echo "removed $label"
    ;;
  status)
    if launchctl list | grep -q "$label"; then echo "$label is loaded ($plist)"; else echo "$label is not loaded"; fi
    if [ -f "$root/data/logs/refresh.log" ]; then
      echo "--- last lines of data/logs/refresh.log"; tail -n 15 "$root/data/logs/refresh.log"
    fi
    ;;
  *)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    [ -z "$1" ] || [ "$1" = -h ] || [ "$1" = --help ] || exit 2
    ;;
esac
