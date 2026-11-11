#!/usr/bin/env bash
# tomorrow.sh: which stocks are expected to gain +0.5% or more in the next
# session, measured from its open, judged from each stock's state at 15:00
# today (or at the latest finished session's 15:00, before the open).
#
#   scripts/tomorrow.sh                  the next session's summary
#   scripts/tomorrow.sh --target 1       a different gain (trains a model for it if needed)
#   scripts/tomorrow.sh --session 2026-11-06   the session after a past day (in-sample if trained on)
#   scripts/tomorrow.sh --evaluate       redo the held-out check behind "how much to trust it"
#
# Takes every option of today.sh; it is today.sh --tomorrow.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/today.sh" --tomorrow "$@"
