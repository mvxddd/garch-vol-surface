#!/bin/bash
# Install the daily snapshot as a macOS launchd agent (the native scheduler;
# cron on macOS needs Full Disk Access and fails silently without it).
#
#   scripts/install_schedule.sh                      # weekdays 17:30 local
#   scripts/install_schedule.sh --time 18:00 --tickers "SPY QQQ"
#   scripts/install_schedule.sh --remove
#   scripts/install_schedule.sh --status
#   scripts/install_schedule.sh --run-now            # fire it once, right now
#
# Idempotent: re-running replaces the agent rather than stacking another.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.volsurface.snapshot"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TEMPLATE="$PROJECT_DIR/scripts/$LABEL.plist.template"
TIME="17:30"
TICKERS="SPY"
ACTION="install"

while [ $# -gt 0 ]; do
    case "$1" in
        --time)    TIME="$2"; shift 2 ;;
        --tickers) TICKERS="$2"; shift 2 ;;
        --remove)  ACTION="remove"; shift ;;
        --status)  ACTION="status"; shift ;;
        --run-now) ACTION="run"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$ACTION" in
  status)
    if launchctl list | grep -q "$LABEL"; then
        echo "Installed and loaded:"
        launchctl list | grep "$LABEL"
        echo "Plist: $PLIST"
    else
        echo "Not installed."
    fi
    exit 0 ;;
  remove)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || \
        launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed $LABEL."
    exit 0 ;;
  run)
    launchctl kickstart -p "gui/$(id -u)/$LABEL" 2>/dev/null || \
        "$PROJECT_DIR/scripts/daily_snapshot.sh"
    echo "Triggered a run. Watch: tail -f $PROJECT_DIR/outputs/logs/snapshot.log"
    exit 0 ;;
esac

HOUR="${TIME%%:*}"; MINUTE="${TIME##*:}"
HOUR="$((10#$HOUR))"; MINUTE="$((10#$MINUTE))"     # strip any leading zero

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/outputs/logs"
sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__TICKERS__|$TICKERS|g" \
    -e "s|__HOUR__|$HOUR|g" \
    -e "s|__MINUTE__|$MINUTE|g" \
    "$TEMPLATE" > "$PLIST"

plutil -lint "$PLIST" >/dev/null || { echo "generated plist is invalid" >&2; exit 1; }

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load "$PLIST"

echo "Installed $LABEL — weekdays at $(printf '%02d:%02d' "$HOUR" "$MINUTE"), tickers: $TICKERS"
echo "  plist:    $PLIST"
echo "  history:  $PROJECT_DIR/outputs/history/"
echo "  log:      $PROJECT_DIR/outputs/logs/snapshot.log"
echo
echo "Check it:   scripts/install_schedule.sh --status"
echo "Test it:    scripts/install_schedule.sh --run-now"
echo "Remove it:  scripts/install_schedule.sh --remove"
