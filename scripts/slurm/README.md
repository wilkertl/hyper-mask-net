# Slurm scripts for the CEIA DGX Spark cluster

Jobs run on DGX Spark nodes (aarch64, one NVIDIA GB10, ~80 GB usable shared CPU/GPU memory,
one exclusive node per job). Paths are hard-coded to `user_wilkerrodrigues`; change them for
another account.

| File | Purpose |
|---|---|
| `import-image.sbatch` | One-off: build `~/images/pytorch-25.09.sqsh` (NGC PyTorch 25.09, ~15 GB) |
| `container.sh` | Unpack the image to node-local `/tmp` (~1 min) and run a script inside it |
| `check.sbatch` / `check.sh` | Smoke test: GPU, torch, repo, internet access from the node |
| `f0-spike.sbatch` / `f0-spike.sh` | F0 spike, `python -m hyperdime.spike all` on the local GPU backend |

## Cluster quirks these scripts work around

- `%u` is not expanded in `--chdir`, so paths are written out.
- `sbatch` on the login node rejects `--container-image` (pyxis), and pyxis unpacking the image
  onto BeeGFS timed out after 15 min; `container.sh` calls enroot directly on local disk instead.
- The login container has no `rsync` and no GNU `parallel` (needed by `enroot import`).
- The shared enroot cache `/beegfs/scratch/enroot/cache` holds other users' unreadable layers.
- `~/.cache` is owned by root and will not change; caches (`XDG_CACHE_HOME`, `HF_HOME`) live in
  the container's node-local `/tmp`, so the model is downloaded again on each job (~1.2 GB).

## Usage

From the desktop, copy the repo (no `rsync` on the cluster):

```bash
tar czf - --exclude=./artifacts --exclude=./data --exclude=./runs --exclude=./reports \
  --exclude=./.venv --exclude='__pycache__' --exclude='.*_cache' . \
  | ssh -p 2222 user_wilkerrodrigues@10.100.20.210 'mkdir -p ~/hyper-mask-net/logs && tar xzf - -C ~/hyper-mask-net'
```

On the login node:

```bash
cd ~/hyper-mask-net
sbatch scripts/slurm/import-image.sbatch   # once
sbatch scripts/slurm/check.sbatch
sbatch scripts/slurm/f0-spike.sbatch
```

Bring results back to the desktop:

```bash
ssh -p 2222 user_wilkerrodrigues@10.100.20.210 'cd ~/hyper-mask-net && tar czf - reports artifacts logs' \
  | tar xzf - -C .
```
