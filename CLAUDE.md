# CLAUDE.md — scPRINT-2 Codebase Guide

## Project Overview

**scPRINT-2** is a single-cell RNA-seq foundation model built by Jérémie Kalfon (Cantini Lab) using PyTorch Lightning. It has been pretrained on 350M+ cells from 22,000+ datasets across 16 species.

### What it does (zero-shot)
- **Expression denoising & imputation** — reconstruct full expression from sparse/noisy data
- **Cell embedding & batch correction** — low-dimensional cell representations
- **Label prediction** — cell type, disease, tissue, sex, age, ethnicity, sequencer
- **Gene regulatory network (GRN) inference** — attention-based network extraction
- **Cross-species integration** — shared gene vocabulary across 16 species

---

## Repository Structure

```
scPRINT-2/
├── scprint2/
│   ├── model/
│   │   ├── model.py        # Main scPRINT2 LightningModule (2713 lines — core of everything)
│   │   ├── encoders.py     # GeneEncoder, ContinuousValueEncoder, EasyExprGNN, etc.
│   │   ├── decoders.py     # ExprDecoder, ClsDecoder, MVCDecoder, VAEDecoder
│   │   ├── loss.py         # ZINB, NB, MSE, contrastive (CCE/ECS), hierarchical cls losses
│   │   ├── fsq.py          # Finite Scalar Quantization for cell embedding compression
│   │   └── utils.py        # WeightedMasker, Attention tracker, test utilities
│   ├── tasks/
│   │   ├── denoise.py      # Denoiser class — inference wrapper for denoising
│   │   ├── cell_emb.py     # Embedder class — cell embedding + classification
│   │   ├── grn.py          # GNInfer class — gene regulatory network inference
│   │   ├── impute.py       # Imputer class
│   │   ├── finetune.py     # FinetuneBatchClass
│   │   ├── gene_emb.py     # GeneEmbeddingExtractor
│   │   ├── generate.py     # Expression generation
│   │   └── tmfg.py         # TMFG graph algorithm for GRN filtering
│   ├── tokenizers/
│   │   ├── embedder.py     # Gene tokenization utilities
│   │   └── protein_embedder.py
│   ├── trainer/
│   │   └── trainer.py      # TrainingMode callback — sets training hyperparams on model
│   ├── utils/
│   │   ├── utils.py        # Misc utilities
│   │   ├── sinkhorn.py     # Sinkhorn distance for GRN
│   │   ├── graph_refinement.py
│   │   └── get_seq.py
│   ├── cli.py              # LightningCLI entry point (MyCLI)
│   ├── datasets.py         # Dataset helpers
│   └── base.py             # Package name constant
├── config/                 # YAML configs for training (base_v2.yml, pretrain_*.yml, etc.)
├── tests/
│   ├── test_base.py        # Single integration test covering all major features
│   └── conftest.py
├── notebooks/              # Usage examples (get-started, GRN, embedding, etc.)
├── slurm/                  # HPC job scripts
├── figures/                # Visualization scripts
├── tools/                  # Geneformer adapters
├── pyproject.toml
└── Makefile
```

---

## Key Components

### 1. Model (`scprint2/model/model.py`)

Main class: `scPRINT2(L.LightningModule, PyTorchModelHubMixin)`

**Architecture flow:**
```
Input: gene indices + raw counts
    ↓
GeneEncoder         — embedding lookup for gene identities
    +
ExprEncoder         — continuous MLP / binned embedding / metacell GNN for expression values
    +
PositionalEncoder   — optional genomic position encoding (sinusoidal over chr positions)
    ↓
Transformer         — FlashTransformer (default) or Performer
                    — Gene tokens + class tokens (cell_type, disease, ...) concatenated
    ↓
ExprDecoder         — predicts ZINB(mean, disp, zero_logits) or MSE per gene
ClsDecoder(s)       — predicts cell type, disease, etc. from class tokens
MVCDecoder          — optional multi-view coding decoder
```

**Multi-species:** genes are stored per-organism in `self._genes` dict; `self.genes` property flattens them in organism order.

**Training objectives (all combined):**
- Masked expression prediction (BERT-style)
- Denoising (dropout of counts then reconstruct)
- Expression generation from cell embeddings
- ECS loss (elastic cell similarity — within-batch cosine regularization)
- CCE loss (cross-view contrastive between augmentations)
- Hierarchical classification loss (BCE with ontology structure)
- Optional: MVC, VAE KL, adversarial batch correction

### 2. TrainingMode (`scprint2/trainer/trainer.py`)

A Lightning `Callback` that sets training hyperparameters on the model at `on_train_start`. This pattern is used because Lightning separates model and training config, but scPRINT2 bundles training flags directly on the model object.

```python
TrainingMode(
    noise=[0.6],           # denoising dropout levels
    mask_ratio=[0.15],     # masking ratios (can include "TF" for TF-weighted masking)
    warmup_duration=500,   # LR warmup steps
    cce_scale=0.2,         # contrastive loss weight
    ecs_scale=0.2,
    class_scale=1.0,
    do_generate=True,      # bottleneck generation task
)
```

### 3. Task Classes (`scprint2/tasks/`)

Each task class is a standalone inference wrapper. Call pattern:
```python
task = Denoiser(batch_size=10, max_len=5000, ...)
result = task(model, adata)
```

- **Denoiser** — denoises AnnData, returns metrics + denoised adata
- **Embedder** — generates cell embeddings + class predictions
- **GNInfer** — extracts GRNs from attention weights (softmax/sinkhorn preprocessing, TMFG/thresh filtration)
- **Imputer** — imputes unmeasured genes

### 4. CLI (`scprint2/cli.py`)

Uses `jsonargparse` + `LightningCLI`. Entry point: `scprint2` command.

```bash
scprint2 fit --config config/base_v2.yml --config config/pretrain_medium.yml
scprint2 denoise --adata my.h5ad --ckpt_path v2-medium.ckpt --species NCBITaxon:9606
scprint2 embed --adata my.h5ad --ckpt_path v2-medium.ckpt
scprint2 gninfer --adata my.h5ad --ckpt_path v2-medium.ckpt --cell_type "T cell"
```

Key CLI link: `data.genes_dict → model.genes` (automatically wired via `link_arguments`)

### 5. Losses (`scprint2/model/loss.py`)

| Function | Purpose |
|---|---|
| `zinb(target, mu, theta, pi)` | Zero-inflated NB (main expression loss) |
| `nb(target, mu, theta)` | Negative binomial (no zero-inflation) |
| `mse(input, target)` | MSE with log1p + normalization |
| `contrastive_loss(x, y, temp)` | NT-Xent / InfoNCE (CCE) |
| `ecs(cell_emb, threshold)` | Elastic cell similarity |
| `hierarchical_classification(pred, cl, hierarchy)` | BCE with ontology parent handling |
| `within_sample(cell_embs)` | Embedding independence regularizer |

---

## Entry Points

### Python API
```python
from scprint2 import scPRINT2
from scprint2.tasks import Denoiser, Embedder, GNInfer

model = scPRINT2.load_from_checkpoint('v2-medium.ckpt', precpt_gene_emb=None, gene_pos_file=None)

denoiser = Denoiser(batch_size=10, max_len=5000)
metrics, indices, adata_denoised = denoiser(model, adata)
```

### Training
```python
from lightning.pytorch import Trainer
from scprint2 import scPRINT2
from scprint2.trainer import TrainingMode
from scdataloader import DataModule

datamodule = DataModule(collection_name="...", ...)
model = scPRINT2(genes=datamodule.genes_dict, organisms=datamodule.organisms, d_model=512, ...)
trainer = Trainer(callbacks=[TrainingMode(noise=[0.6], mask_ratio=[0.15])])
trainer.fit(model, datamodule=datamodule)
```

---

## Dev Setup

```bash
git clone https://github.com/cantinilab/scPRINT-2
cd scPRINT-2
uv venv env --python 3.11
source env/bin/activate
uv pip install -e ".[dev]"

# Required: init lamin
lamin init --storage ./testdb --name testdb --modules bionty
scprint2 easy_setup  # populates ontologies + downloads default checkpoint
```

### Running Tests

```bash
pytest tests/test_base.py -v
```

Note: tests require lamindb setup, a valid `tests/test.h5ad`, and will download `small-v2.ckpt` from HuggingFace on first run. Only one test function (`test_base`) covers denoising, embedding, GRN, and training — no unit tests for individual components.

---

## Key Dependencies

| Package | Role |
|---|---|
| `lightning` | Training framework |
| `lamindb` / `bionty` | Ontology management, gene/cell metadata |
| `scdataloader` | DataModule, Collator, Preprocessor for scRNA data |
| `simpler_flash` | FlashTransformer implementation |
| `performer_pytorch` | Linear attention alternative |
| `grnndata` | GRN-annotated AnnData format |
| `bengrn` | GRN benchmarking |
| `scanpy` / `anndata` | Single-cell data handling |
| `huggingface_hub` | Checkpoint download/upload (PyTorchModelHubMixin) |

---

## Important Conventions

- **Gene vocabulary** stored as `dict[organism_id, list[gene_name]]` in `self._genes`; access via `self.genes` property (flattened)
- **Class tokens** prepended to gene sequence in transformer; positions `0..len(classes)` are cell embeddings, rest are gene tokens
- **`on_load_checkpoint`** is heavily patched for backward compatibility — old model keys are remapped there
- **Training flags** (noise, mask_ratio, etc.) are instance attributes set by `TrainingMode` callback, NOT constructor args
- **`predict_step` → `_predict`** accumulates results in `self.embs`, `self.pred`, `self.pos`, `self.expr_pred` across batches; flushed to disk at `max_size_in_mem` cells
- **Attention extraction** via `get_attention_layer` — returns QKV tensors from specified layers, used for GRN inference

---

## Task Classes — Additional Notes

### Embedder (`tasks/cell_emb.py`)
- Clean implementation, mirrors Denoiser pattern
- **Note:** calls `model.log_adata()` which writes to disk — requires `model.logger.save_dir` to be set or falls back to `/tmp`/`data/`. Make sure a logger is attached before calling Embedder in production.
- `keep_all_labels_pred=True` accumulates `self.pred` in CPU memory — can OOM on very large datasets

### FinetuneBatchClass (`tasks/finetune.py`)
- Complex finetuning for batch correction / new species. Works via gradient descent on frozen backbone.
- **[STUB]** `FinetuneGRN`, `FinetuneGeneEmb`, `FinetuneNewClass`, `FinetuneUpdateClass` are all empty `pass` classes — not implemented at all.

### Imputer (`tasks/impute.py`)
- Two modes: `"masking"` (mask target genes, predict) and `"generative"` (generate from cell embedding)
- Straightforward, no major issues found

### EasyExprGNN (`model/encoders.py`)
- Non-deepset GNN path has a `# TODO: to finish` comment — the `neighbors` branch in the GNN forward pass is incomplete. If `gnn_type != "deepset"`, neighbors go through `gnn_layer` but are never combined with `x` — they're computed and discarded. **[BUG]** This path is silently broken.

### protein_embeddings_generator (`tokenizers/embedder.py`)
- RNA embedding section is commented out with `# TODO: to redebug` — ncRNA embeddings are not supported despite the code skeleton existing.

---

## Issues Found

1. **[BUG] `model/model.py`~L1027** — Normalization `"both"` is broken. The code uses `if/elif` chain:
   ```python
   if self.normalization in ["sum", "both"]:
       expression = expression / expression.sum(1).unsqueeze(1)
   elif self.normalization in ["log", "both"]:  # ← DEAD BRANCH for "both"
       expression = torch.log2(1 + expression)
   ```
   When `normalization="both"`, the `elif` is never reached. The docstring says "both: Sum normalization then log transform" but only sum is applied. Fix: use `if/if` or restructure.

2. **[BUG] `model/model.py`~L700** — `add_organism()` with memmap gene encoder: code prints `"todev.. will fail for now"` and continues, which will cause a crash downstream when trying to modify the memmap embedding. This is an incomplete implementation left in production code.

3. **[BUG] `model/model.py`~L2694 (`on_predict_epoch_end`)** — `if self.pos.shape[0] < 100` will raise `AttributeError` if `self.pos` is `None` (which happens when memory limit is hit and pos is reset to `None` mid-epoch). Should guard with `if self.pos is None: return`.

4. **[BUG] `model/model.py`~L2233 (`validation_step`)** — Same issue: `if self.embs is not None: if self.pos.shape[0] < 100_000` — `self.pos` can be `None` even when `self.embs` is not `None`.

5. **[WARNING] `model/model.py`~L1384** — `bias` variable is only assigned inside `if self.attn_bias is not None:` block but referenced later as `bias if self.attn_bias is not None else None`. This works because of the re-check, but if the block is ever refactored it becomes a `NameError`. Minor: initialize `bias = None` at the top.

6. **[WARNING] `tasks/grn.py`** — Imports `from bengrn import BenGRN, get_perturb_gt, get_sroy_gt` and `from bengrn.base import train_classifier`. These private module paths are fragile and will break if `bengrn` is updated. No version pin enforced.

7. **[TODO] `model/model.py`~L1423** — `if do_next_tp: pass` — next time-point prediction (task 7) is a stub with no implementation.

8. **[WARNING] `model/model.py`~L2003** — `loss.zinb(..., mask=self.mask_zeros)` passes a boolean to `zinb()`. Inside `zinb`, when `mask=True`, it computes `mask = (target > 0).float()`. This is intentional but misleading — the local `mask_zeros` tensor (the actual attention mask) is never passed to the loss function. Confusing naming between the bool flag and the tensor.

9. **[WARNING] `tests/test_base.py`** — Single monolithic test function. Any failure aborts all subsequent checks with no isolation. Should be split into `test_denoising`, `test_embedding`, `test_grn`, `test_training`.

10. **[WARNING] `model/model.py`~L519** — `if finetune_gene_emb and False:` — dead code block that will never execute. The `and False` disables an entire branch permanently.
11. **[BUG] `model/encoders.py`~L668** — `EasyExprGNN` non-deepset GNN path: `neighbors` tensor is processed through `gnn_layer` but never merged with `x` — result silently discarded. Only the `deepset` path works correctly.
12. **[STUB] `tasks/finetune.py`~L560-575** — `FinetuneGRN`, `FinetuneGeneEmb`, `FinetuneNewClass`, `FinetuneUpdateClass` are all empty classes (`pass`). Not implemented.
13. **[TODO] `tokenizers/embedder.py`~L82** — ncRNA embedding path commented out (`# TODO: to redebug`). RNA gene embeddings not supported.
14. **[WARNING] `tasks/cell_emb.py`** — `Embedder.__call__` calls `model.log_adata()` which writes to disk. Requires an attached Lightning logger or falls back to `data/` directory. Can fail silently in standalone inference.
