#!/bin/bash
#SBATCH --job-name=f0-check
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45-ephem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/beegfs/home/user_wilkerrodrigues/hyper-mask-net
#SBATCH -o /beegfs/home/user_wilkerrodrigues/hyper-mask-net/logs/%x-%j.out
#SBATCH -e /beegfs/home/user_wilkerrodrigues/hyper-mask-net/logs/%x-%j.err
bash /beegfs/home/user_wilkerrodrigues/hyper-mask-net/scripts/slurm/container.sh /beegfs/home/user_wilkerrodrigues/hyper-mask-net/scripts/slurm/check.sh
