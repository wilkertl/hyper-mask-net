#!/bin/bash
#SBATCH --job-name=f0-spike
#SBATCH --account=default
#SBATCH --partition=hpc-spark-45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --chdir=/beegfs/home/user_wilkerrodrigues/hyper-mask-net
#SBATCH -o /beegfs/home/user_wilkerrodrigues/hyper-mask-net/logs/%x-%j.out
#SBATCH -e /beegfs/home/user_wilkerrodrigues/hyper-mask-net/logs/%x-%j.err
bash /beegfs/home/user_wilkerrodrigues/hyper-mask-net/scripts/slurm/container.sh /beegfs/home/user_wilkerrodrigues/hyper-mask-net/scripts/slurm/f0-spike.sh
