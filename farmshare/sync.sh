#!/bin/bash
# Sync the working tree to FarmShare (no data, venvs, or runs) and stamp the code version.
set -euo pipefail
cd "$(dirname "$0")/.."
socket="${FARMSHARE_SOCKET:-/tmp/farmshare-litsearch.sock}"
host="${FARMSHARE_HOST:-kwx04@login.farmshare.stanford.edu}"
sha="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain -- src configs farmshare)" ]]; then sha="${sha}-dirty"; fi
rsync -az --exclude .git --exclude .venv --exclude artifacts --exclude runs \
  --exclude __pycache__ --exclude '*.pyc' --exclude GIT_SHA \
  -e "ssh -S $socket" ./ "$host:~/literature-search-agent/"
ssh -S "$socket" "$host" "echo $sha > ~/literature-search-agent/GIT_SHA"
echo "synced $sha"
