#!/bin/bash
#SBATCH --job-name=dca_pancreas_fix
#SBATCH --output=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_pancreas_fix_%j.out
#SBATCH --error=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_pancreas_fix_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common

echo "=== Fix DCA pancreas started on $(hostname) at $(date) ==="

DCA_PYTHON="/pasteur/appa/homes/jkalfon/scPRINT/.dca-env/bin/python3"
SCRIPT="/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/fix_pancreas_dca.py"

$DCA_PYTHON $SCRIPT

echo "=== Done at $(date) ==="
