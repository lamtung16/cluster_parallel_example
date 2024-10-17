#!/bin/bash
#SBATCH --array=0-215
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/tln229/simple_cluster_demo/slurm-out/slurm-%A_%a.out
#SBATCH --error=/scratch/tln229/simple_cluster_demo/slurm-out/slurm-%A_%a.out
#SBATCH --job-name=simple_cluster_demo

cd /scratch/tln229/simple_cluster_demo
python 2_run_one.py $SLURM_ARRAY_TASK_ID
