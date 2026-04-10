#!/bin/bash
#SBATCH --job-name=dca_step1_prep
#SBATCH --output=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_step1_%j.out
#SBATCH --error=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_step1_%j.err
#SBATCH --time=00:45:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=common

echo "=== Step1 (scPRINT env, no GPU) started on $(hostname) at $(date) ==="

PYTHON="/pasteur/appa/scratch/jkalfon/scprint_test/bin/python3"
SCRIPT="/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/step1_prepare_subadata.py"

$PYTHON $SCRIPT

echo "=== Done at $(date) ==="
