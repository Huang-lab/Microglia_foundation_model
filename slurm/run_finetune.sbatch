#!/bin/bash
#SBATCH --job-name=mfm_finetune
#SBATCH --partition=gpu                 # <-- set to the correct Minerva GPU partition
#SBATCH --gres=gpu:a100:1               # one A100 80GB
#SBATCH --cpus-per-task=8               # match finetune num_workers
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/sc/arion/scratch/%u/mfm/logs/%x_%j.out
#SBATCH --error=/sc/arion/scratch/%u/mfm/logs/%x_%j.err
# #SBATCH --account=<your_alloc>        # uncomment + set if Minerva requires it

set -euo pipefail

# --- 0. Paths (edit) -------------------------------------------------------
SCRATCH="/sc/arion/scratch/${USER}/mfm"
CKPT="${SCRATCH}/checkpoints/small-v2.ckpt"     # pre-staged on a login node
ADATA="${SCRATCH}/data/microglia_census.h5ad"   # pre-staged Census cache
export LAMIN_CACHE_DIR="${SCRATCH}/lamin"        # pre-populated ontology DB
export HF_HOME="${SCRATCH}/hf"                   # offline HF cache
export HF_HUB_OFFLINE=1                          # compute node assumed offline
mkdir -p "${SCRATCH}/logs"

# --- 1. Environment --------------------------------------------------------
module purge
module load anaconda3            # <-- match Minerva's module names
module load cuda                 # <-- match the CUDA your torch build expects
source activate mfm              # the project conda env (built on a GPU node)

# --- 2. Sanity check before the long run ----------------------------------
python scripts/verify_env.py --ckpt "${CKPT}"

# --- 3. Fine-tune ----------------------------------------------------------
# Replace with the real scprint2 CLI invocation / training entrypoint once the
# Phase 4 config is written. Example skeleton:
#   scprint2 fit --config config/base_v2.yml \
#       --model.ckpt_path "${CKPT}" \
#       --data.adata "${ADATA}"
echo "[mfm] env OK on $(hostname); wire the scprint2 fit/finetune command here."

echo "[mfm] done."
