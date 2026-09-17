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
    # "Loaded" is not "working". This agent sat loaded for four days while
    # every run died on a macOS permission error, and nothing said so — the
    # only visible symptom was a history store that quietly stopped growing.
    # So report what actually happened, not what is registered.
    if ! launchctl list | grep -q "$LABEL"; then
        echo "NOT INSTALLED"
        exit 1
    fi
    line="$(launchctl list | grep "$LABEL")"
    last_exit="$(echo "$line" | awk '{print $2}')"
    echo "loaded:        yes"
    echo "plist:         $PLIST"
    if [ "$last_exit" = "0" ] || [ "$last_exit" = "-" ]; then
        echo "last exit:     $last_exit (ok)"
    else
        echo "last exit:     $last_exit  <-- FAILING"
        case "$last_exit" in
          126) echo "               126 means the script could not be executed." ;;
          127) echo "               127 means the script or interpreter was not found." ;;
        esac
        [ -s "$PROJECT_DIR/outputs/logs/launchd.err.log" ] && {
            echo "               last error:"
            tail -2 "$PROJECT_DIR/outputs/logs/launchd.err.log" | sed 's/^/                 /'
        }
    fi

    log="$PROJECT_DIR/outputs/logs/snapshot.log"
    if [ -s "$log" ]; then
        echo "last log line: $(tail -1 "$log")"
    else
        echo "last log line: (the script has never produced output)"
    fi

    # The number that actually matters: is the history still growing?
    "$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR" <<'PYEOF' 2>/dev/null ||         echo "newest snapshot: (could not read the history store)"
import sys, pathlib
sys.path.insert(0, sys.argv[1])
from volsurface.history import SurfaceHistory
store = SurfaceHistory(pathlib.Path(sys.argv[1]) / "outputs" / "history")
import datetime as dt
found = False
for path in sorted(store.directory.glob("*_surface.parquet")):
    ticker = path.name.replace("_surface.parquet", "").replace("_", "^")
    s = store.summary(ticker)
    if s.get("n_snapshots"):
        age = (dt.date.today() - dt.date.fromisoformat(s["last"])).days
        flag = "" if age <= 4 else f"   <-- {age} days stale"
        print(f"history:       {ticker}: {s['n_snapshots']} snapshot(s), "
              f"newest {s['last']}{flag}")
        found = True
if not found:
    print("history:       empty")
PYEOF
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

# Prove launchd can actually run it, rather than trusting that "loaded" means
# working. On macOS a Desktop/Documents/Downloads path is TCC-protected and a
# launchd agent is refused with "Operation not permitted" — installed, loaded,
# and dead. Better to fail here than to discover it from a history store that
# stopped growing a week ago.
echo
echo "Verifying launchd can execute the job..."
launchctl kickstart -p "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
sleep 3
status_line="$(launchctl list | grep "$LABEL" || true)"
last_exit="$(echo "$status_line" | awk '{print $2}')"
if [ "$last_exit" = "126" ] || [ "$last_exit" = "1" ] &&    grep -q "Operation not permitted" "$PROJECT_DIR/outputs/logs/launchd.err.log" 2>/dev/null
then
    echo
    echo "  PROBLEM: launchd cannot execute the job (exit $last_exit)."
    grep "Operation not permitted" "$PROJECT_DIR/outputs/logs/launchd.err.log" \
        | tail -1 | sed 's/^/    /'
    echo
    echo "  This project lives under a macOS-protected folder (Desktop,"
    echo "  Documents or Downloads). Scheduled jobs are denied access there."
    echo "  Two fixes:"
    echo "    1. Move the project somewhere unprotected, e.g. ~/projects/, and"
    echo "       re-run this installer. Cleanest — no system permissions needed."
    echo "    2. Grant Full Disk Access to /bin/bash in System Settings >"
    echo "       Privacy & Security. Broader than it needs to be."
    echo
    echo "  The agent is installed but will not run until one of those is done."
    exit 1
fi
echo "  ok (exit ${last_exit:-unknown})"
echo
echo "Check it:   scripts/install_schedule.sh --status"
echo "Test it:    scripts/install_schedule.sh --run-now"
echo "Remove it:  scripts/install_schedule.sh --remove"
