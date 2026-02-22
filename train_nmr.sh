#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu=64G
#SBATCH --no-requeue

IMG=/home/software/singularity/pytorch.simg\:2026-02-04

cd ~/nmr
apptainer exec --nv $IMG python3 train_nmr.py


