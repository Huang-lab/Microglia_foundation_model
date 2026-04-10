#!/bin/bash
#SBATCH --job-name=scprint2_dca
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_slurm_%j.out
#SBATCH --error=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_slurm_%j.err

set -e
echo "=== DCA job started on $(hostname) at $(date) ==="

PYTHON=/pasteur/appa/scratch/jkalfon/scprint_test/bin/python3
cd /pasteur/appa/homes/jkalfon/scPRINT
export MPLBACKEND=Agg

$PYTHON dca_benchmark.py

echo "=== Done at $(date) ==="
