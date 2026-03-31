#!/bin/bash
set -euo pipefail

if [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  USER_HOME="$HOME"
fi

APP_DIR="$USER_HOME/aries-multi-channel-demo/build"
APP_BIN="$APP_DIR/src/demo/demo"

cd "$APP_DIR"

if [ ! -x "$APP_BIN" ]; then
  echo "Demo binary not found or not executable: $APP_BIN"
  echo "Run ./update.sh first to build the project."
  exit 1
fi

"$APP_BIN"
