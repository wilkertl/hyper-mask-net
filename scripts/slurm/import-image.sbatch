#!/bin/bash
#SBATCH --job-name=import-image
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45-ephem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/beegfs/home/user_wilkerrodrigues
#SBATCH -o /beegfs/home/user_wilkerrodrigues/hyper-mask-net/logs/%x-%j.out
#SBATCH -e /beegfs/home/user_wilkerrodrigues/hyper-mask-net/logs/%x-%j.err
# Run once: the login container lacks GNU parallel, and the shared enroot cache holds other
# users' unreadable layers, so import on a node with a private node-local cache.
set -euo pipefail
hostname; uname -m
mkdir -p "$HOME/images"
export ENROOT_CACHE_PATH=/tmp/$USER/enroot-cache ENROOT_TEMP_PATH=/tmp/$USER/enroot-tmp
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"
trap 'rm -rf /tmp/$USER' EXIT
rm -f "$HOME/images/pytorch-25.09.sqsh"
enroot import -a aarch64 -o "$HOME/images/pytorch-25.09.sqsh" 'docker://nvcr.io#nvidia/pytorch:25.09-py3'
ls -lh "$HOME/images"
