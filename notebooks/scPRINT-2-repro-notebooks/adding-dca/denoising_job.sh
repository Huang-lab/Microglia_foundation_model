#!/bin/bash
#SBATCH --job-name=scprint2_denoising
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/slurm_%j.out
#SBATCH --error=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/slurm_%j.err

set -e
echo "=== Job started on $(hostname) at $(date) ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"

mkdir -p /pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results

PYTHON=/pasteur/appa/scratch/jkalfon/scprint_test/bin/python3
SCRIPT=/pasteur/appa/homes/jkalfon/scPRINT/denoising_benchmark.py

echo "Python: $PYTHON"
$PYTHON --version

cd /pasteur/appa/homes/jkalfon/scPRINT
export MPLBACKEND=Agg

echo "=== Running denoising benchmark ==="
$PYTHON $SCRIPT

echo "=== Job finished at $(date) ==="
