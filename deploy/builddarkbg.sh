#!/bin/bash
# (Re)build and (re)start the DarkBGHelmet engine on the VPS.
#
# Copy this file to the home folder (like buildzettel.sh) and run it:
#   ./builddarkbg.sh
#
# 1. stops the pm2 process (if running)
# 2. clones or updates ~/nodejs/DarkBGHelmet from the public GitHub repo
# 3. builds bgserver and bgcli (C++20, optimised for this CPU)
# 4. starts/restarts the pm2 process "bgengine" on 127.0.0.1:$BG_PORT
#
# One-time prerequisites (needs sudo):
#   sudo apt update && sudo apt install -y build-essential cmake git
set -euo pipefail

APP=bgengine
DIR="$HOME/nodejs/DarkBGHelmet"
REPO=https://github.com/kagsteiner/DarkBGHelmet.git
BRANCH=master
export BG_PORT="${BG_PORT:-3050}"
export BG_THREADS="${BG_THREADS:-2}"
export BG_HASH_MB="${BG_HASH_MB:-64}"
export BG_PLIES="${BG_PLIES:-3}"

# pm2 lives in the nvm-managed Node installation; load nvm for non-interactive shells.
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
command -v pm2 >/dev/null || { echo "pm2 not found (nvm not loaded?)"; exit 1; }

for tool in g++ cmake git; do
  command -v $tool >/dev/null || { echo "$tool missing: sudo apt install -y build-essential cmake git"; exit 1; }
done

echo "== 1. stop $APP"
pm2 stop "$APP" 2>/dev/null || true

echo "== 2. sync $DIR with $REPO ($BRANCH)"
if [ ! -d "$DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"
git fetch --tags origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"   # the server copy is never edited by hand
echo "   at $(git log --oneline -1)"

echo "== 3. build"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" --target bgserver bgcli
mkdir -p data
./build/bgcli move 4HPwATDgc/ABMA 31 2 >/dev/null   # smoke test; creates data/bearoff_os15.bin on first run
echo "   build ok"

echo "== 4. start $APP on 127.0.0.1:$BG_PORT ($BG_THREADS threads)"
if pm2 describe "$APP" >/dev/null 2>&1; then
  pm2 delete "$APP" >/dev/null   # re-read the config (port/threads may have changed)
fi
pm2 start deploy/ecosystem.config.js
pm2 save >/dev/null

sleep 1
if curl -fs "http://127.0.0.1:$BG_PORT/v1/health" >/dev/null; then
  echo "== $APP is up: $(curl -s "http://127.0.0.1:$BG_PORT/v1/info" | grep -o '"network": *"[^"]*"')"
else
  echo "== $APP did not answer yet; check: pm2 logs $APP"
fi
