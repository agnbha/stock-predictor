#!/usr/bin/env bash
# Runs Grafana for Stock Predictor on http://localhost:3001 (admin/admin), beside stock-analyzer's on 3000.
#
# Reuses the Grafana build and SQLite plugin stock-analyzer's grafana/run-local.sh downloaded, when
# they are there (../stock-analyzer/grafana/.runtime); otherwise downloads its own into
# grafana/.runtime/ (about 100 MB, once). Its settings, dashboards' state and logs live in
# grafana/.runtime/ either way: nothing is installed system-wide, and nothing of stock-analyzer's is
# changed. Delete grafana/.runtime/ to undo it all. Ctrl-C stops it.
#
#   grafana/run-local.sh                  start it
#   GRAFANA_PORT=3002 grafana/run-local.sh another port
set -euo pipefail

GRAFANA_VERSION="${GRAFANA_VERSION:-11.6.0}"
GRAFANA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GRAFANA_DIR/.." && pwd)"
RUNTIME="$GRAFANA_DIR/.runtime"
PORT="${GRAFANA_PORT:-3001}"
PREDICTOR_DB="${STOCK_PREDICTOR_DB:-$REPO_ROOT/data/stock-predictor.db}"
ANALYZER_DB="${STOCK_ANALYZER_DB:-$REPO_ROOT/../stock-analyzer/data/stock-analyzer.db}"
SIBLING="$REPO_ROOT/../stock-analyzer/grafana/.runtime"

case "$(uname -m)" in
  arm64) ARCH="darwin-arm64" ;;
  x86_64) ARCH="darwin-amd64" ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

for db in "$PREDICTOR_DB" "$ANALYZER_DB"; do
  if [[ ! -f "$db" ]]; then
    echo "No database at $db" >&2
    echo "Run scripts/today.sh once (stock-predictor's), or set STOCK_PREDICTOR_DB / STOCK_ANALYZER_DB." >&2
    exit 1
  fi
done
PREDICTOR_DB="$(cd "$(dirname "$PREDICTOR_DB")" && pwd)/$(basename "$PREDICTOR_DB")"
ANALYZER_DB="$(cd "$(dirname "$ANALYZER_DB")" && pwd)/$(basename "$ANALYZER_DB")"

mkdir -p "$RUNTIME"

# The Grafana build: stock-analyzer's if it has one, else our own.
HOME_PATH="$SIBLING/grafana-v$GRAFANA_VERSION"
if [[ ! -f "$HOME_PATH/conf/defaults.ini" ]]; then
  HOME_PATH="$RUNTIME/grafana-v$GRAFANA_VERSION"
  if [[ ! -f "$HOME_PATH/conf/defaults.ini" ]]; then
    echo "Downloading Grafana $GRAFANA_VERSION for $ARCH (about 100 MB, once)..."
    curl -fSL --progress-bar "https://dl.grafana.com/oss/release/grafana-$GRAFANA_VERSION.$ARCH.tar.gz" \
      -o "$RUNTIME/grafana.tar.gz"
    tar xzf "$RUNTIME/grafana.tar.gz" -C "$RUNTIME"
    rm -f "$RUNTIME/grafana.tar.gz"
  fi
fi
if [[ ! -f "$HOME_PATH/conf/defaults.ini" ]]; then
  echo "Grafana did not unpack where expected: $HOME_PATH" >&2
  echo "Delete $RUNTIME and re-run, or set GRAFANA_VERSION to a version that exists." >&2
  exit 1
fi

# The SQLite plugin: copied from stock-analyzer's if it has it, else installed.
PLUGINS="$RUNTIME/plugins"
mkdir -p "$PLUGINS" "$RUNTIME/data" "$RUNTIME/logs" "$RUNTIME/provisioning/datasources" \
  "$RUNTIME/provisioning/dashboards"
if [[ ! -d "$PLUGINS/frser-sqlite-datasource" ]]; then
  if [[ -d "$SIBLING/plugins/frser-sqlite-datasource" ]]; then
    cp -R "$SIBLING/plugins/frser-sqlite-datasource" "$PLUGINS/"
  else
    echo "Installing the SQLite datasource plugin..."
    "$HOME_PATH/bin/grafana" cli --homepath "$HOME_PATH" --pluginsDir "$PLUGINS" \
      plugins install frser-sqlite-datasource
  fi
fi

# The committed provisioning has placeholders; fill in the real paths.
sed -e "s#__STOCK_PREDICTOR_DB__#$PREDICTOR_DB#" -e "s#__STOCK_ANALYZER_DB__#$ANALYZER_DB#" \
  "$GRAFANA_DIR/provisioning/datasources/sqlite.yaml" > "$RUNTIME/provisioning/datasources/sqlite.yaml"
sed "s#__DASHBOARDS__#$GRAFANA_DIR/dashboards#" \
  "$GRAFANA_DIR/provisioning/dashboards/dashboards.yaml" > "$RUNTIME/provisioning/dashboards/dashboards.yaml"

cat > "$RUNTIME/custom.ini" <<INI
[paths]
data = $RUNTIME/data
logs = $RUNTIME/logs
plugins = $PLUGINS
provisioning = $RUNTIME/provisioning

[server]
http_port = $PORT

[security]
admin_user = admin
admin_password = admin

[users]
default_theme = dark

[dashboards]
default_home_dashboard_path = $GRAFANA_DIR/dashboards/01-picks-per-stock.json

[analytics]
reporting_enabled = false
check_for_updates = false
INI

echo
echo "Grafana starting on http://localhost:$PORT  (admin / admin)"
echo "Dashboards are in the 'Stock Predictor' folder. Ctrl-C to stop."
echo "Reading: $PREDICTOR_DB"
echo "         $ANALYZER_DB (read-only)"
echo
exec "$HOME_PATH/bin/grafana" server --homepath "$HOME_PATH" --config "$RUNTIME/custom.ini"
