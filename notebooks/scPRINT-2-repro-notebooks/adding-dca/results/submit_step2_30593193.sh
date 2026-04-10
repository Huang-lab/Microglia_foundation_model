#!/bin/bash
#SBATCH --job-name=dca_step2_dca
#SBATCH --output=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_step2_%j.out
#SBATCH --error=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_step2_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common
#SBATCH --dependency=afterok:30593193

echo "=== Step2 (DCA env) started on $(hostname) at $(date) ==="

PYTHON="/pasteur/appa/homes/jkalfon/scPRINT/.dca-env/bin/python3"
SCRIPT="/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/step2_run_dca.py"

$PYTHON $SCRIPT

echo "=== Done at $(date) ==="
