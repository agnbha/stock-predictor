#!/usr/bin/env bash
# Adds the Stock Predictor dashboards to stock-analyzer's Grafana, as a "Stock Predictor" folder beside its
# "Stock Analyzer" one: one Grafana (http://localhost:3000), no second instance.
#
#   grafana/add-to-stock-analyzer.sh            add (or update) them
#   grafana/add-to-stock-analyzer.sh remove     take them out again
#   grafana/add-to-stock-analyzer.sh status     whether they are there
#
# stock-analyzer's grafana/run-local.sh keeps Grafana's provisioning in its grafana/.runtime/provisioning/
# and rewrites only its own two files there on each start. This writes two more, both named
# stock-predictor.yaml: the datasources (this project's store, and stock-analyzer's database read without
# writing) and the dashboards folder, which points at this repo's grafana/dashboards. Nothing of
# stock-analyzer's own changes. Run stock-analyzer's grafana/run-local.sh once first, so the runtime exists;
# run this again if its grafana/.runtime is ever deleted.
#
# A running Grafana loads the new files when asked to reload, which needs its admin login:
#   GRAFANA_PASSWORD=... grafana/add-to-stock-analyzer.sh     (GRAFANA_USER, default admin; GRAFANA_URL,
#                                                              default http://localhost:3000)
# Without it, restart stock-analyzer's Grafana (its grafana/run-local.sh) to pick them up.
set -euo pipefail

GRAFANA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GRAFANA_DIR/.." && pwd)"
ANALYZER_ROOT="${STOCK_ANALYZER_ROOT:-$REPO_ROOT/../stock-analyzer}"
ANALYZER_ROOT="$(cd "$ANALYZER_ROOT" 2>/dev/null && pwd || echo "$ANALYZER_ROOT")"
RUNTIME="${STOCK_ANALYZER_GRAFANA_RUNTIME:-$ANALYZER_ROOT/grafana/.runtime}"
PREDICTOR_DB="${STOCK_PREDICTOR_DB:-$REPO_ROOT/data/stock-predictor.db}"
ANALYZER_DB="${STOCK_ANALYZER_DB:-$ANALYZER_ROOT/data/stock-analyzer.db}"
URL="${GRAFANA_URL:-http://localhost:3000}"
USER_="${GRAFANA_USER:-admin}"
DS_FILE="$RUNTIME/provisioning/datasources/stock-predictor.yaml"
DB_FILE="$RUNTIME/provisioning/dashboards/stock-predictor.yaml"

abs() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }

reload() {
  if [ -z "${GRAFANA_PASSWORD:-}" ]; then
    echo "Restart stock-analyzer's Grafana (its grafana/run-local.sh) to load the change, or run this again"
    echo "with GRAFANA_PASSWORD=<its admin password> to have the running one reload."
    return
  fi
  local ok=1 kind code
  for kind in datasources dashboards; do
    code="$(curl -s -o /dev/null -w '%{http_code}' -u "$USER_:$GRAFANA_PASSWORD" -X POST \
      "$URL/api/admin/provisioning/$kind/reload" || true)"
    [ "$code" = 200 ] || { ok=0; echo "Reloading $kind at $URL answered $code."; }
  done
  if [ "$ok" = 1 ]; then echo "Grafana at $URL reloaded."; else
    echo "Check GRAFANA_URL / GRAFANA_USER / GRAFANA_PASSWORD, or restart stock-analyzer's Grafana."; fi
}

case "${1:-add}" in
  add)
    if [ ! -d "$RUNTIME/provisioning" ]; then
      echo "No Grafana runtime at $RUNTIME." >&2
      echo "Run stock-analyzer's grafana/run-local.sh once first (or set STOCK_ANALYZER_GRAFANA_RUNTIME)." >&2
      exit 1
    fi
    for db in "$PREDICTOR_DB" "$ANALYZER_DB"; do
      [ -f "$db" ] || { echo "No database at $db (set STOCK_PREDICTOR_DB / STOCK_ANALYZER_DB)." >&2; exit 1; }
    done
    sed -e "s#__STOCK_PREDICTOR_DB__#$(abs "$PREDICTOR_DB")#" -e "s#__STOCK_ANALYZER_DB__#$(abs "$ANALYZER_DB")#" \
      "$GRAFANA_DIR/provisioning/datasources/stock-predictor.yaml" > "$DS_FILE"
    sed "s#__DASHBOARDS__#$GRAFANA_DIR/dashboards#" \
      "$GRAFANA_DIR/provisioning/dashboards/stock-predictor.yaml" > "$DB_FILE"
    echo "Added to stock-analyzer's Grafana: $DS_FILE"
    echo "                                   $DB_FILE"
    echo "Dashboards: the 'Stock Predictor' folder, beside 'Stock Analyzer', at $URL"
    reload
    ;;
  remove)
    rm -f "$DS_FILE" "$DB_FILE"
    echo "Removed the Stock Predictor provisioning from $RUNTIME/provisioning."
    reload
    ;;
  status)
    if [ -f "$DS_FILE" ] && [ -f "$DB_FILE" ]; then echo "Added: $DS_FILE, $DB_FILE"; else echo "Not added."; fi
    ;;
  -h|--help)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    ;;
  *)
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2
    ;;
esac
