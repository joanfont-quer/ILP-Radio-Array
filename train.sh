#!/bin/bash
#SBATCH --job-name=ILP-Radio-Array
#SBATCH --time=10-23:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2_3/jfont/ILP-Radio-Array/slurm_logs/slurm_%j.out
#SBATCH --array=1

pwd

export GRB_LICENSE_FILE=/share/nas2_3/jfont/.gurobi/wls.lic
echo "Using license file: $GRB_LICENSE_FILE"

echo ">>> Activating environment"
source /share/nas2_3/jfont/miniconda3/etc/profile.d/conda.sh
conda activate ILP-Radio-Array-3.10

echo ">>> Starting job $SLURM_ARRAY_TASK_ID"
python /share/nas2_3/jfont/ILP-Radio-Array/main.py