#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
source "$ROOT/scripts/lib/load_dotenv.zsh"
load_dotenv_file "$ROOT/.env"
load_dotenv_file "$ROOT/.env.policy"

LABEL="com.zotero-local-rag.nightly"
TEMPLATE="$ROOT/scripts/$LABEL.plist.example"
TARGET="${NIGHTLY_PLIST_PATH:-$HOME/Library/LaunchAgents/$LABEL.plist}"
START_TIME="${NIGHTLY_START_TIME:-01:00}"
LAUNCH_MODE="${NIGHTLY_LAUNCH_MODE:-direct}"

if [[ "$START_TIME" != [0-9][0-9]:[0-9][0-9] ]]; then
  echo "Invalid NIGHTLY_START_TIME: $START_TIME (expected HH:MM)" >&2
  exit 2
fi
START_HOUR="${START_TIME%%:*}"
START_MINUTE="${START_TIME##*:}"
if (( 10#$START_HOUR > 23 || 10#$START_MINUTE > 59 )); then
  echo "Invalid NIGHTLY_START_TIME: $START_TIME (expected 00:00-23:59)" >&2
  exit 2
fi
if [[ "$LAUNCH_MODE" != "direct" && "$LAUNCH_MODE" != "terminal" ]]; then
  echo "Invalid NIGHTLY_LAUNCH_MODE: $LAUNCH_MODE (expected direct or terminal)" >&2
  exit 2
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "nightly schedule: $START_TIME local time"
  echo "nightly launch mode: $LAUNCH_MODE"
  echo "launchd target: $TARGET"
  if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
    echo "launchd service: loaded"
  else
    echo "launchd service: not loaded"
  fi
  exit 0
fi

mkdir -p "${TARGET:h}"
TMP="$(mktemp "${TARGET:h}/.$LABEL.XXXXXX")"
trap 'rm -f "$TMP"' EXIT
cp "$TEMPLATE" "$TMP"
/usr/libexec/PlistBuddy -c "Set :StartCalendarInterval:Hour $((10#$START_HOUR))" "$TMP"
/usr/libexec/PlistBuddy -c "Set :StartCalendarInterval:Minute $((10#$START_MINUTE))" "$TMP"
if [[ "$LAUNCH_MODE" == "terminal" ]]; then
  LAUNCH_LOG_DIR="$HOME/Library/Logs"
  mkdir -p "$LAUNCH_LOG_DIR"
  /usr/libexec/PlistBuddy -c "Set :ProgramArguments:0 /usr/bin/open" "$TMP"
  /usr/libexec/PlistBuddy -c "Add :ProgramArguments:1 string -a" "$TMP"
  /usr/libexec/PlistBuddy -c "Add :ProgramArguments:2 string Terminal" "$TMP"
  /usr/libexec/PlistBuddy -c "Add :ProgramArguments:3 string $ROOT/scripts/nightly_summaries.sh" "$TMP"
  /usr/libexec/PlistBuddy -c "Set :WorkingDirectory $HOME" "$TMP"
  /usr/libexec/PlistBuddy -c "Set :StandardOutPath $LAUNCH_LOG_DIR/zotero-local-rag-nightly-launchd.log" "$TMP"
  /usr/libexec/PlistBuddy -c "Set :StandardErrorPath $LAUNCH_LOG_DIR/zotero-local-rag-nightly-launchd.log" "$TMP"
else
  /usr/libexec/PlistBuddy -c "Set :ProgramArguments:0 $ROOT/scripts/nightly_summaries.sh" "$TMP"
  /usr/libexec/PlistBuddy -c "Set :WorkingDirectory $ROOT" "$TMP"
  /usr/libexec/PlistBuddy -c "Set :StandardOutPath $ROOT/data/nightly_launchd.log" "$TMP"
  /usr/libexec/PlistBuddy -c "Set :StandardErrorPath $ROOT/data/nightly_launchd.log" "$TMP"
fi
plutil -lint "$TMP" >/dev/null
chmod 600 "$TMP"
mv "$TMP" "$TARGET"
trap - EXIT

DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "$DOMAIN" "$TARGET"
launchctl enable "$DOMAIN/$LABEL"
echo "Installed $LABEL at $START_TIME local time (mode=$LAUNCH_MODE): $TARGET"
