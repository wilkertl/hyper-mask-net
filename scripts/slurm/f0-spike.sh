#!/bin/bash
set -euo pipefail
export HOME=/beegfs/home/user_wilkerrodrigues
# ~/.cache on the cluster is owned by root and not writable; keep caches node-local
export XDG_CACHE_HOME=/tmp/cache HF_HOME=/tmp/cache/huggingface
mkdir -p "$HF_HOME"
unset V4_EMBEDDING_BASE_URL
cd "$HOME/hyper-mask-net"

# venv on top of the container's CUDA torch; --no-deps keeps pip from replacing it
VENV=$HOME/venvs/hyperdime
if [ ! -x "$VENV/bin/python" ]; then
  python -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install --no-deps -e .
  "$VENV/bin/pip" install "transformers>=4.51"
fi
PY=$VENV/bin/python

nvidia-smi -L
$PY -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
$PY -m hyperdime.spike all --backend local --device cuda --batch-size 256
