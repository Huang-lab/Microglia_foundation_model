# SETUP.md — Reproducible environment (Phase 1)

This repo targets **two** compute environments. Pick the one you are on.

| Target | Hardware | Tooling | Status |
|--------|----------|---------|--------|
| **A. Local dev box (GB10)** | NVIDIA GB10 (Grace-Blackwell, **arm64/aarch64**, sm_121), 120 GB unified mem | `uv` | ✅ verified green |
| **B. Minerva HPC (Mount Sinai)** | NVIDIA A100 80 GB (**x86_64**), SLURM | `module load` + conda, `sbatch` | ⚙️ parameterized, untested from here |

The package, checkpoint loading, and `tests/test_base.py` all pass on **A**.
Path **B** keeps the upstream x86 dependency story intact (see notes below).

---

## A. Local dev box — GB10 / arm64 (`uv`)

### Prereqs
- `uv` (tested with 0.11.16)
- NVIDIA driver supporting **CUDA ≥ 12.9** (verified: driver 580.142 / CUDA 13.0)
- Python is provisioned by `uv` (project pins 3.11; range `>=3.10,<3.13`)

### 1. Sync the environment
```bash
uv sync            # core runtime deps
uv sync --extra dev   # adds pytest, ruff, mkdocs, etc. (needed to run tests)
```

### 2. Why the arm64 path needs special handling
PyPI does **not** ship a CUDA-enabled `torch==2.8.0` wheel for aarch64 (only
CPU-only). Three surgical fixes live in `pyproject.toml` (`[tool.uv]`), all
scoped by platform marker so the x86/Minerva path is untouched:

1. **torch / torchvision / torchaudio / triton → cu129 index** on aarch64.
   `download.pytorch.org/whl/cu129` ships `torch 2.8.0+cu129` aarch64 wheels
   whose sm_120 kernels run forward-compatibly on the GB10's sm_121.
2. **`triton` pinned to 3.3.0** on aarch64 (stock torch 2.8 wants 3.4.0, which
   is x86-only on PyPI; cu129 ships 3.3.0 for aarch64).
3. **`torchtext` scoped to x86_64 only.** It is EOL upstream, unused in this
   codebase, and has no torch-2.8-compatible aarch64 wheel. It is declared only
   as a transitive dep of `esm` and `simpler_flash`; dropping it on arm64 is
   safe.

Verified working stack on GB10:
`torch 2.8.0+cu129`, `cuda 12.9`, `triton 3.3.0`, Python 3.11.15.

> The CUDA capability warning (`GB10 ... capability 12.1 ... max supported 12.0`)
> is cosmetic — real GPU matmul and the full training smoke test pass.

### 3. Data files not in git (`data/` is `.gitignore`d)
A fresh clone is missing runtime data files. Fetch from upstream:
```bash
mkdir -p data/main
curl -sL -o data/main/TFs.txt \
  https://raw.githubusercontent.com/cantinilab/scPRINT-2/main/data/main/TFs.txt
# test fixtures
curl -sL -o tests/test.h5ad \
  https://raw.githubusercontent.com/cantinilab/scPRINT-2/main/tests/test.h5ad
curl -sL -o tests/test_emb.parquet \
  https://raw.githubusercontent.com/cantinilab/scPRINT-2/main/tests/test_emb.parquet
```

### 4. Stand up lamin.ai + ontologies (hard dependency)
```bash
uv run lamin init --storage ./mfm-lamindb --name mfm-scprint --modules bionty
```
The test suite's `conftest.py` also creates a throwaway `test-scprint` instance
and `populate_my_ontology()` downloads the gene / cell-type / organism / disease
ontologies on first run (slow once, cached after).

### 5. Diagnostic + green baseline
```bash
uv run python scripts/verify_env.py          # expect 0 FAIL
uv run python scripts/verify_env.py --ckpt tests/small-v2.ckpt   # after a ckpt exists
uv run pytest tests/test_base.py -x           # full GPU smoke test, ~4-5 min
```
Expected: `verify_env.py` → 0 FAIL; `pytest` → `1 passed`.

---

## B. Minerva HPC — A100 / x86_64 (conda + SLURM)

On x86_64 the `pyproject.toml` platform markers fall back to the normal PyPI
resolution (stock `torch==2.8.0`, `triton==3.4.0`, `torchtext` included), so the
arm64 overrides above do **not** apply.

### 1. Build the env on a GPU node (CUDA present, so flash-attn builds for A100)
```bash
module purge
module load anaconda3      # <-- match Minerva's actual module names
module load cuda           # <-- match the CUDA your torch build expects
conda create -n mfm python=3.11 -y
conda activate mfm
pip install -e ".[dev,flash]"   # flash extra builds simpler_flash[flash] against the A100
```
> Build the env on a **GPU node**, not a login node, so flash-attention compiles
> against the A100 (sm_80).

### 2. Pre-stage everything on a login/transfer node → scratch
Compute nodes may have no egress. Stage before submitting:
- checkpoint → `$SCRATCH/mfm/checkpoints/small-v2.ckpt`
- lamin ontology DB → `$SCRATCH/mfm/lamin`
- Census `.h5ad` cache → `$SCRATCH/mfm/data/`
- HF cache → `$SCRATCH/mfm/hf` (`export HF_HUB_OFFLINE=1` on compute nodes)
- `data/main/TFs.txt` (see A.3)

### 3. Verify + run via SLURM
```bash
srun --partition=gpu --gres=gpu:a100:1 --pty bash    # interactive A100
python scripts/verify_env.py --ckpt $SCRATCH/mfm/checkpoints/small-v2.ckpt
pytest tests/test_base.py -x
```
For long jobs use the template: `slurm/run_finetune.sbatch` (edit partition,
account, module names, and the scratch paths at the top).

---

## Quick reference — green gates
- `scripts/verify_env.py` → **0 FAIL**
- `pytest tests/test_base.py -x` → **1 passed**

Once both pass, the environment is ready for Phase 2 (data assembly) / Phase 3
(zero-shot baseline).
