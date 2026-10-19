#!/bin/bash
# Runs one full notebook on Khipu (used for notebooks 06, 07 and 08).
# Submit from ~/tp53-transfer-learning/khipu:
#   sbatch --mail-type=END,FAIL --mail-user="$CORREO" notebook.slurm 06_network_part_A.ipynb
# If it stops (24 h limit), submit it again. The notebooks continue from the last saved part.

#SBATCH --job-name=tp53_nb
#SBATCH --account=tesis
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out

set -eo pipefail
export LD_PRELOAD="${LD_PRELOAD:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"  # PyTorch uses the 32 cores for each network

module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate tp53dl

cd "$HOME/tp53-transfer-learning/notebooks"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$1"
echo "Done: $1"
