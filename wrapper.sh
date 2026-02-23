#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu=64G
#SBATCH --no-requeue
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2026-02-04

cd ~/nmr
apptainer exec --env KERAS_BACKEND=torch --env PYTHONUNBUFFERED=1 --nv $IMG "$@"
