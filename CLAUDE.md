# Microglia_foundation_model — CLAUDE.md

> Project memory for Claude / Claude Code. Keep this file current; it is the
> first thing an agent should read.

## What this project is

A **microglia-specialized single-cell foundation model**, built as a fork of
[scPRINT-2](https://github.com/cantinilab/scPRINT-2) (Cantini Lab). The goal is
**specialization, not pretraining from scratch**: start from a released
scPRINT-2 checkpoint and fine-tune it into a microglia-state expert
(homeostatic ↔ disease-associated / DAM, neuroinflammation, neurodegeneration
contexts).

The underlying architecture, class name (`scPRINT2`), and importable package
(`scprint2`) are **kept from upstream** so checkpoints load and upstream fixes
merge cleanly. Only project *identity* (repo URLs, docs, this file) is forked.

## Repo layout (unchanged from upstream)

- `scprint2/model/` — encoders, transformer, decoders, FSQ quantizer, losses
- `scprint2/tasks/` — `finetune`, `cell_emb`, `grn`, `denoise`, `impute`,
  `generate`, `gene_emb`
- `scprint2/tokenizers/` — ESM3 (protein-coding) + RNABert (non-coding) gene
  embedders
- `scprint2/trainer/`, `scprint2/cli.py` — Lightning + `jsonargparse` CLI
  (`scprint2 ...`)
- `config/` — training/pretrain/predict YAMLs
- `tests/` — `test_base.py` runs against `small-v2.ckpt`

## Environment — Minerva HPC (Mount Sinai)

- **Scheduler:** SLURM. Interactive work via `salloc`/`srun`; long runs via
  `sbatch` (never foreground on a login node).
- **GPU target:** NVIDIA A100 80 GB. Request explicitly, e.g.
  `--gres=gpu:a100:1` on the appropriate GPU partition. 80 GB comfortably fits
  fine-tuning small/medium (and larger batch/`max_len` than a 24 GB card).
- **Toolchain:** Lmod modules + conda. `module load` the CUDA + anaconda stack,
  then a project conda env (Python 3.10–3.12, `torch==2.8.0`). Flash-attention
  via the `flash` extra (`simpler_flash`) — build on a GPU node with CUDA loaded.
- **lamin.ai is a hard dependency** (gene / cell-type / organism ontologies).
- ⚠️ **Compute nodes likely have no/limited internet.** Pre-stage on a
  login/data-transfer node and read offline in jobs: (1) conda/pip env build,
  (2) HuggingFace checkpoint download, (3) lamin ontology population, (4)
  CELLxGENE Census query → cached `.h5ad`. **Verify Minerva's egress policy** and
  adjust. This is the most common pipeline-breaker on HPC.

## Storage (Minerva)

- **Code / env:** home or project space (`scprint2/` package + this repo).
- **Data / checkpoints / outputs:** scratch (`/sc/arion/scratch/<user>/...`).
  Mind scratch **purge policies** — keep nothing irreplaceable there.
- Record the staged checkpoint path + Census cache path per experiment.

## Conventions

- **Do not rename** the `scprint2` package/import, the `scPRINT2` class, the
  acronym expansion in docstrings, or the Lightning error-string match in
  `model.py`. These are functional / compatibility-critical.
- Checkpoints: fine-tune from a released small/medium checkpoint; record which
  one in each experiment. (Note: a corrupt stub checkpoint exists upstream — do
  not use unverified `.ckpt` files; verify load before training.)
- `origin` = your fork; `upstream` = `cantinilab/scPRINT-2` for pulling fixes.

## Current status

- [ ] Phase 0 — fork hygiene & framing
- [ ] Phase 1 — reproducible env + green test baseline
- [ ] Phase 2 — microglia data assembled
- [ ] Phase 3 — zero-shot baseline + eval harness
- [ ] Phase 4 — fine-tuning + ablations
- [ ] Phase 5 — biological validation
- [ ] Phase 6 — packaged showcase

See `PROJECT_PLAN.md` for the full plan and `PHASE0_RENAME.md` for the rename
changeset.

## Next steps

1. Run the Phase 0 changeset; rewrite identity files; set remotes.
2. Build the conda env (`module load` CUDA + anaconda), install the package +
   `flash` extra on a GPU node, then green `tests/test_base.py` via an
   interactive `srun` on an A100.
3. Pre-stage checkpoint + lamin ontologies + Census cache to scratch.
4. Assemble the microglia corpus and define the state-label schema.
