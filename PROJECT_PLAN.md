# Microglia_foundation_model — Project Plan

> A microglia-specialized single-cell foundation model, built on
> [scPRINT-2](https://github.com/cantinilab/scPRINT-2). This repository is a
> fork whose goal is **specialization, not pretraining from scratch**: take a
> generalist 16-species / 350M-cell model and turn it into a microglia-state
> expert.

This document is both the engineering roadmap and a demonstration of an
efficient human + Claude workflow across software engineering, deep learning,
and computational biology. Each phase names the **deliverable** and the
**collaboration pattern** (which work goes to agentic Claude Code, which to
high-reasoning models, which to local models under a cost-aware routing policy).

## Scope & non-goals

- **In scope:** repo specialization, reproducible env, microglia data assembly,
  zero-shot baseline, fine-tuning/specialization on the small/medium checkpoint,
  biological validation, packaging.
- **Out of scope:** full pretraining of the large model from scratch (350M cells
  does not fit on a single A5000-class GPU). We fine-tune from a released
  checkpoint.
- **Compute target:** Minerva HPC (Mount Sinai), **LSF (`bsub`)**, **NVIDIA B200**
  (Blackwell sm_100), conda/module-managed env, `torch==2.8.0` **built for CUDA
  12.9 (cu129)** — stock PyPI x86_64 torch lacks sm_100 kernels. Fine-tuning
  small/medium fits comfortably; full pretrain does not. Long runs go through
  `bsub` (`jobs/finetune.lsf`). A second dev target, an NVIDIA GB10
  (Grace-Blackwell, arm64), is verified green for local iteration.

## Phase status

| Phase | Title | Domain | Status |
|------:|-------|--------|--------|
| 0 | Fork hygiene & project framing | SWE | ☐ |
| 1 | Reproducible environment | SWE / DevOps | ◑ (GB10 green; Minerva parameterized) |
| 2 | Microglia data assembly | Comp bio | ☐ |
| 3 | Zero-shot baseline | DL + comp bio | ☐ |
| 4 | Specialization / fine-tuning | DL | ☐ |
| 5 | Biological validation & downstream | Comp bio | ☐ |
| 6 | Package the showcase | SWE / comms | ☐ |

Phases 0–3 are a credible standalone showcase. Phases 4–5 are the headline
(real scientific contribution + the most interesting Claude collaboration).

---

## Phase 0 — Fork hygiene & project framing (software engineering)

Reset the project's identity without breaking the architecture's identity.

- [ ] Clone, set `upstream` remote to `cantinilab/scPRINT-2`, `origin` to your
      fork, so architecture fixes can be pulled later.
- [ ] Replace `CLAUDE.md` with project memory for *this* fork (provided).
- [ ] Reframe `README.md` H1 + intro as "microglia-specialized model built on
      scPRINT-2"; fix repo-URL badges/links to the fork.
- [ ] Update identity-only references: `mkdocs.yml` (`site_name`, `site_url`),
      `pyproject.toml` `[project.urls].repository`, docs deploy URL in
      `Makefile`, `docs/index.md` repo URLs, `CONTRIBUTING.md`.
- [ ] **Keep functional, do not rename:** the `scprint2` package dir and import
      name, the `scPRINT2` class, the acronym expansion in docstrings, the
      Lightning error-string match in `model.py`, and the `scprint2` CLI/script
      entry point. This preserves checkpoint loading and clean upstream merges.

See `PHASE0_RENAME.md` for the exact, reviewable changeset.

**Claude pattern:** agentic Claude Code inventories every reference and proposes
one reviewable diff; you stay in the review seat. The judgment call (identity vs.
architecture name) is exactly what makes this more than blind find/replace.

## Phase 1 — Reproducible environment (software engineering / DevOps)

- [x] **GB10 (arm64) dev path with `uv`** — green. `uv sync` + `uv sync --extra
      dev`; resolved the arm64/Blackwell torch story (see `SETUP.md`): cu129
      CUDA wheels for torch/vision/audio, `triton==3.3.0`, and `torchtext`
      scoped to x86_64 only (EOL + unused). `torch 2.8.0+cu129`, CUDA available
      on `NVIDIA GB10` (sm_121, fwd-compat).
- [x] **Minerva facts verified (2026-06-13):** scheduler is **LSF** (not SLURM);
      GPUs are **B200 / sm_100** (not A100); modules `cuda/12.9.1`, `gcc/12.2.0`,
      `anaconda3/2025.06` present; queues `gpu`/`gpuexpress`/`interactive`;
      `bsub -P acc_huangk06a` mandatory; compute nodes **have egress**.
- [ ] **Minerva (x86_64 B200) conda path** — scripted but env build not yet run:
      `jobs/create_cond_env.sh` (loads modules, installs **cu129 torch** then
      `pip install -e ".[dev,flash]"`, builds flash-attn **on a GPU node**).
- [x] Egress confirmed on compute nodes → full pre-staging optional; cache
      checkpoint / lamin ontology DB / Census `.h5ad` to scratch as a courtesy.
- [x] Stand up lamin.ai (ontologies for genes / cell types / organisms) —
      `lamin init ... --modules bionty`; ontologies populate on first test run.
- [x] Run `tests/test_base.py` → **green baseline on GB10** (full
      Preprocessor→Denoiser→Embedder→GNInfer→train pipeline, `1 passed`).
      Still TODO via interactive `bsub -Is` on a B200.
- [x] Capture working setup in `SETUP.md` (both paths) + **LSF** template
      (`jobs/finetune.lsf`); `slurm/run_finetune.sbatch` deprecated (wrong
      scheduler), `jobs/{create,activate}_conda_env.sh` corrected.
- [ ] Recover the `data/`-gitignored runtime files on each fresh clone
      (`data/main/TFs.txt`, `tests/test.h5ad`, `tests/test_emb.parquet`) — see
      `SETUP.md` A.3.

**Claude pattern:** route the "make the suite pass on this node" traceback loop
to an agentic coding tool; review the module/conda pins.

## Phase 2 — Microglia data assembly (computational biology)

- [ ] Choose corpus spanning homeostatic + disease-associated states (e.g. human
      brain immune atlases via CELLxGENE; mouse 5xFAD/DAM for the canonical
      homeostatic→DAM trajectory).
- [ ] Build a `scDataLoader`-compatible AnnData collection with required `obs`
      keys (`organism_ontology_term_id`, `cell_type_ontology_term_id`, a `batch`
      key) via the repo `Preprocessor`.
- [ ] Define the microglia state label set you will predict.

**Claude pattern:** high-reasoning Claude for dataset selection, ontology-term
reconciliation, and label-schema design — lit synthesis, not code-grinding.

## Phase 3 — Zero-shot baseline (deep learning + comp bio)

Run the *pretrained* model before any training to quantify what specialization
buys you.

- [ ] `Embedder` → cell embeddings + label predictions + UMAP; does the
      generalist already separate states / correct batch?
- [ ] `Denoiser` and `GNInfer` (GRN) on a microglia subset.
- [ ] Eval harness: scIB batch/bio-conservation, label F1, GRN recovery vs. a
      known microglia TF set (e.g. SPI1, IRF8, MEF2C).

**Claude pattern:** Claude drafts the analysis notebook and the metric code; this
becomes the "before" panel.

## Phase 4 — Specialization / fine-tuning (deep learning) — core

Use `tasks/finetune.py` (`FinetuneBatchClass`): supports `predict_keys`, batch
learning, MMD batch correction, `xpressor` fine-tune mode.

- [ ] Fine-tune small/medium checkpoint to predict microglia states + integrate
      across donors/conditions.
- [ ] Ablations (fits the A5000 budget): ± MMD, ± `learn_batches_on`, frozen vs.
      unfrozen encoder.
- [ ] Optional focused experiment: FSQ-bottleneck behavior on a narrow cell-type
      manifold.

**Claude pattern:** architecture/loss reasoning + ablation design → high-reasoning
model; the "launch run → parse W&B → summarize → propose next hyperparameter"
loop → cheaper local models per the routing policy. A clean cost-aware
multi-agent demonstration on a real DL task.

## Phase 5 — Biological validation & downstream (computational biology)

Show the specialized model does what the generalist does not.

- [ ] Recover homeostatic→DAM transition + marker programs (TREM2, APOE, CST7).
- [ ] Microglia-specific GRN; compare hub TFs to literature.
- [ ] Quantify lift over the Phase 3 baseline on identical metrics.

**Claude pattern:** Claude as validation partner — interpret GRNs against known
biology, flag confident-but-wrong predictions, draft the results narrative.

## Phase 6 — Package the showcase (software engineering + communication)

- [ ] One reproducible end-to-end notebook (zero-shot → fine-tune → validate).
- [ ] CI smoke test (tiny version of the pipeline).
- [ ] Refreshed README with a results figure + short write-up.

**Claude pattern:** Claude as documentation/presentation engineer for the final
deliverable.

---

## Risks & notes

- **Don't blanket-rename `scPRINT-2`.** It is both project identity *and* the
  architecture's scientific name. Surgical edits only (Phase 0).
- **Checkpoint compatibility:** keep the `scPRINT2` class name; `load_from_checkpoint`
  depends on it.
- **lamin.ai is a hard dependency** for ontology resolution — budget setup time.
- **B200 (~180 GB HBM3e):** ample for fine-tuning small/medium; scale
  `max_len`/`batch_size` up before reaching for gradient checkpointing. Full
  pretrain still out of scope. Requires the cu129 torch build (sm_100).
- **HPC egress:** pre-stage all downloads (checkpoint, lamin, Census) on a
  login/transfer node; compute nodes may be offline.
