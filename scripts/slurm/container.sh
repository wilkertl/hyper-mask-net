#!/bin/bash
# Usage: container.sh <script> — unpack the image to node-local /tmp, run <script> inside it.
set -euo pipefail
H=/beegfs/home/user_wilkerrodrigues
IMAGE=$H/images/pytorch-25.09.sqsh
NAME=pytorch-${SLURM_JOB_ID:-manual}
LOCAL=/tmp/$USER-enroot
export ENROOT_CACHE_PATH=$LOCAL/cache ENROOT_DATA_PATH=$LOCAL/data ENROOT_TEMP_PATH=$LOCAL/tmp
export ENROOT_RUNTIME_PATH=$LOCAL/run
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH"
trap 'enroot remove -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$LOCAL"' EXIT

echo "== $(date +%T) host $(hostname), /tmp free: $(df -h /tmp | awk 'NR==2{print $4}')"
df -hT /tmp | tail -1
echo "== $(date +%T) enroot create (unpacking $(du -h "$IMAGE" | cut -f1) image)"
enroot create --name "$NAME" "$IMAGE"
echo "== $(date +%T) enroot start $1"
enroot start --rw \
  --mount "$H:$H" \
  --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  "$NAME" bash "$1"
echo "== $(date +%T) done"
