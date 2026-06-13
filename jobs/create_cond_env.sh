#!/bin/bash
set -euo pipefail

# Create the Minerva conda env for scprint2 and install the package.
# RUN THIS ON A GPU NODE (so flash-attn builds against the B200), e.g.:
#   bsub -P acc_huangk06a -q gpuexpress -gpu "num=1" -n 8 -W 2:00 -Is bash jobs/create_cond_env.sh

# --- Modules / shell setup ---
module purge
module load anaconda3/2025.06
module load cuda/12.9.1     # B200 (Blackwell sm_100) requires CUDA >= 12.8
module load gcc/12.2.0      # toolchain for source builds (flash-attn)

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths (edit if needed) ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/scprint2"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"
REPO="${REPO:-$(pwd)}"     # run from the repo root

# --- Keep installs off $HOME and avoid user-site leakage ---
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --- Create env if missing (python 3.11 to match the local/verified stack) ---
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create --prefix "${ENV_PREFIX}" python=3.11 -y
fi

# --- Activate env (critical before any pip/conda ops) ---
conda activate "${ENV_PREFIX}"
cd "${REPO}"

# --- Install the package ---------------------------------------------------
# B200 is sm_100 (Blackwell): stock PyPI x86_64 torch 2.8 lacks sm_100 kernels and
# will fail at runtime. Install the CUDA 12.9 build from the PyTorch index FIRST,
# then install the package (with dev + flash extras) without letting pip downgrade torch.
pip install --upgrade pip

pip install --index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"

# Install the project + extras. flash extra builds simpler_flash[flash] against the B200.
pip install -e ".[dev,flash]"

# --- Green gate -----------------------------------------------------------
python scripts/verify_env.py || {
  echo "[mfm] verify_env.py reported FAILs — inspect above before using this env." >&2
  exit 1
}

echo "[mfm] conda env ready at ${ENV_PREFIX}"
