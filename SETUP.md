# SETUP.md — Reproducible environment (Phase 1)

This repo targets **two** compute environments. Pick the one you are on.

| Target | Hardware | Tooling | Status |
|--------|----------|---------|--------|
| **A. Local dev box (GB10)** | NVIDIA GB10 (Grace-Blackwell, **arm64/aarch64**, sm_121), 120 GB unified mem | `uv` | ✅ verified green |
| **B. Minerva HPC (Mount Sinai)** | NVIDIA **B200** (Blackwell sm_100, **x86_64**) | `module load` + conda, **LSF `bsub`** | ⚙️ verified modules/queues, env build untested |

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

## B. Minerva HPC — B200 / x86_64 (conda + LSF `bsub`)

**Verified Minerva facts (2026-06-13):**
- Scheduler is **LSF** (`bsub`/`bjobs`/`bqueues`), *not* SLURM.
- GPUs are **NVIDIA B200** (Blackwell, **sm_100**), *not* A100. They need CUDA
  ≥12.8 and a **cu128/cu129 torch build** — stock PyPI x86_64 torch 2.8 lacks
  sm_100 kernels and fails at runtime.
- Modules confirmed present: `cuda/12.9.1`, `gcc/12.2.0`, `anaconda3/2025.06`.
- GPU queues: `gpu` (max 144h), `gpuexpress` (max 15h), `interactive`.
- **`bsub -P <project>` is mandatory.** Use `-P acc_huangk06a`.
- GPU request syntax: `-q gpu -gpu "num=1"`. Memory is **per-slot MB**:
  `-n 8 -R "rusage[mem=8000]"` ≈ 64 GB total.
- Compute nodes **have internet egress** (HF/PyTorch reachable) — full
  pre-staging is optional; caching is just a courtesy to avoid re-downloads.
- Scratch: `/sc/arion/scratch/$USER` (arion has ~120 TB free). Lab env dir:
  `/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/scprint2`.

### 1. Build the env on a GPU node (so flash-attn builds against the B200)
```bash
# interactive GPU shell on the express queue:
bsub -P acc_huangk06a -q gpuexpress -gpu "num=1" -n 8 -W 2:00 -Is bash
# then, from the repo root:
bash jobs/create_cond_env.sh
```
`jobs/create_cond_env.sh` loads the right modules, creates the conda env
(python 3.11), installs **cu129 torch first** (critical for sm_100), then
`pip install -e ".[dev,flash]"`, and runs `verify_env.py` as a green gate.

> Build on a **GPU node**, not a login node, so flash-attention compiles against
> the B200.

### 2. (Optional) cache big artifacts to scratch
Egress works on compute nodes, so this is optional but avoids re-downloads:
- checkpoint → `$SCRATCH/mfm/checkpoints/small-v2.ckpt`
- lamin ontology DB → `$SCRATCH/mfm/lamin`
- Census `.h5ad` cache → `$SCRATCH/mfm/data/`
- HF cache → `$SCRATCH/mfm/hf`
- `data/main/TFs.txt` (see A.3)

### 3. Verify + run via LSF
```bash
# one-shot GPU diagnostic as a batch job (B200 node, real CUDA checks):
bsub < jobs/verify_env.lsf
# optionally test-load a checkpoint:
CKPT=$SCRATCH/mfm/checkpoints/small-v2.ckpt bsub < jobs/verify_env.lsf

# interactive sanity check on a GPU node:
bsub -P acc_huangk06a -q gpuexpress -gpu "num=1" -n 8 -W 1:00 -Is bash
source jobs/activate_conda_env.sh
python scripts/verify_env.py --ckpt $SCRATCH/mfm/checkpoints/small-v2.ckpt
pytest tests/test_base.py -x

# batch job for long runs:
bsub < jobs/finetune.lsf          # edit -W / mem / the training command first
```

> `jobs/verify_env.lsf` is the Minerva analogue of running `verify_env.py`
> locally: it lands on a **B200 GPU node** so `torch.cuda.is_available()`, GPU
> name/memory, and optional checkpoint loading are exercised for real (a
> login-node run cannot see a GPU). Results land in
> `/sc/arion/scratch/$USER/mfm/logs/verify_<jobid>.out`.

---

## Quick reference — green gates
- `scripts/verify_env.py` → **0 FAIL**
- `pytest tests/test_base.py -x` → **1 passed**

Once both pass, the environment is ready for Phase 2 (data assembly) / Phase 3
(zero-shot baseline).
