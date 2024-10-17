import pandas as pd
import os
import shutil
from datetime import datetime
from itertools import product

# CREATE PARAMETERS CSV
# Define hyperparameters
num_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
layer_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
activation = ['relu', 'tanh']

# Create parameter grid
param_combinations = list(product(num_layers, layer_size, activation))

# Create DataFrame and save it into csv
params_df = pd.DataFrame(param_combinations, columns=['num_layers', 'layer_size', 'activation'])
params_df.to_csv("params.csv", index=False)

# CREATE RUN_ONE.SH
# Define job parameters
n_tasks, ncol = params_df.shape
job_name = "simple_cluster_demo"
job_dir = f"/scratch/tln229/{job_name}"
results_dir = os.path.join(job_dir, "results")

# Create directories
os.makedirs(results_dir, exist_ok=True)

# Create SLURM script
run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output={job_dir}/slurm-out/slurm-%A_%a.out
#SBATCH --error={job_dir}/slurm-out/slurm-%A_%a.out
#SBATCH --job-name={job_name}

cd {job_dir}
python 2_run_one.py $SLURM_ARRAY_TASK_ID
"""

# Write the SLURM script to a file
run_one_sh = os.path.join(job_dir, "run_one.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)