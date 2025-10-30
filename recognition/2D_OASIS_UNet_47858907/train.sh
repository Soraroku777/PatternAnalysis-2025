#!/bin/bash
# SLURM job for training the 2D OASIS U-Net on the Rangpur A100 partition.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --job-name=torch_dawn_bench
#SBATCH -o train.out  
#SBATCH --time=02:00:00
conda activate a3
python train.py
