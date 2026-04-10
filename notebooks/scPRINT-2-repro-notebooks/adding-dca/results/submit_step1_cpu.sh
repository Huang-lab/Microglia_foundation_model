#!/bin/bash
#SBATCH --job-name=dca_step1_prep
#SBATCH --output=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_step1_%j.out
#SBATCH --error=/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_step1_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=common

echo "=== Step1 (scPRINT env, no denoiser) started on vps-56a56e7a at Wed Mar 18 13:27:13 UTC 2026 ==="
/pasteur/appa/scratch/jkalfon/scprint_test/bin/python3 /pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/step1_prepare_subadata.py
echo "=== Done at Wed Mar 18 13:27:13 UTC 2026 ==="
