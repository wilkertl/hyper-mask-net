#!/bin/bash
set -euo pipefail
export HOME=/beegfs/home/user_wilkerrodrigues
# ~/.cache on the cluster is owned by root and not writable; keep caches node-local
export XDG_CACHE_HOME=/tmp/cache
cd "$HOME/hyper-mask-net"

# Same venv as f0-spike.sh; the editable install picks up the current checkout.
VENV=$HOME/venvs/hyperdime
if [ ! -x "$VENV/bin/python" ]; then
  python -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install --no-deps -e .
  "$VENV/bin/pip" install "transformers>=4.51"
fi
PY=$VENV/bin/python

# Diagnostics run on CPU from the cached embeddings and selectors (research R20).
$PY -m hyperdime.spike diagnose
$PY -m hyperdime.spike report
