#!/bin/bash
# Daily surface snapshot.
#
# Installed as a cron job by scripts/install_cron.sh. Appends one row per
# trading day to outputs/history/, which is what turns the anomaly screen from
# cross-sectional ("this tenor is off the curve") into time-series ("this is
# steeper than 97% of the last year") — and what signal_backtest needs.
#
# Deliberately quiet and forgiving: a missed day is a gap in a time series, not
# an emergency, so a failure logs and exits rather than alerting.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
TICKERS="${VOLSURFACE_TICKERS:-SPY}"
LOG_DIR="$PROJECT_DIR/outputs/logs"
LOG="$LOG_DIR/snapshot.log"

mkdir -p "$LOG_DIR"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

# Weekends have no new quotes: the chain is Friday's, and storing it again
# would put a duplicate observation into every z-score. US holidays still slip
# through — the store replaces same-day rows, so a stale repeat is harmless.
DOW="$(date +%u)"
if [ "$DOW" -ge 6 ]; then
    echo "$(date '+%F %T') skipped: weekend" >> "$LOG"
    exit 0
fi

for TICKER in $TICKERS; do
    echo "$(date '+%F %T') snapshot $TICKER ..." >> "$LOG"
    if "$PYTHON" "$PROJECT_DIR/scripts/run_pipeline.py" \
        --ticker "$TICKER" \
        --snapshot \
        --no-walk-forward \
        --no-figures \
        --no-fallback \
        --quiet \
        --out "$PROJECT_DIR/outputs" >> "$LOG" 2>&1
    then
        echo "$(date '+%F %T') snapshot $TICKER ok" >> "$LOG"
    else
        # --no-fallback means a feed outage fails loudly rather than writing a
        # synthetic row into a store meant to hold real market history.
        echo "$(date '+%F %T') snapshot $TICKER FAILED (see above)" >> "$LOG"
    fi
done

# Keep the log from growing without bound.
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 5000 ]; then
    tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
