"""
Step 1 (scPRINT env): preprocess each dataset + filter on model.genes & highly_variable.
No Denoiser. Saves subadata with:
  - layers["true"] = raw counts (full gene set filtered)
  - X = downsampled 0.7 (noisy)
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from scdataloader import Preprocessor

from scprint2 import scPRINT2
from scprint2.model.utils import downsample_profile

# ── Config ────────────────────────────────────────────────────────────────────
LOC = "/pasteur/appa/scratch/jkalfon/data/spcrint_data/temp/"
LOC2 = "/pasteur/appa/homes/jkalfon/scPRINT/models/"
CKPT_CANDIDATES = [
    os.path.join(LOC2, "18hebyht-final-small.ckpt"),
    "/pasteur/appa/homes/jkalfon/scPRINT/tests/small-v2.ckpt",
    "/pasteur/appa/homes/jkalfon/scPRINT/tests/small.ckpt",
]
OUT_DIR = "/pasteur/appa/scratch/jkalfon/data/spcrint_data/dca_subadata/"

os.makedirs(OUT_DIR, exist_ok=True)

datasets = {
    "retina": "https://datasets.cellxgene.cziscience.com/53bd4177-79c6-40c8-b84d-ff300dcf1b5b.h5ad",
    "kidney": "https://datasets.cellxgene.cziscience.com/01bc7039-961f-4c24-b407-d535a2a7ba2c.h5ad",
    "pancreas": "https://figshare.com/ndownloader/files/24539828",
    "intestine": "https://datasets.cellxgene.cziscience.com/d9a99b4a-3755-47c4-8eb5-09821ffbde17.h5ad",
    "glio_smart_highdepth": "https://datasets.cellxgene.cziscience.com/6ec440b4-542a-4022-ac01-56f812e25593.h5ad",
    "lung_smart": "https://datasets.cellxgene.cziscience.com/6ebba0e0-a159-406f-8095-451115673a2c.h5ad",
    "SRX24486462": None,
    "SRX22526970": None,
}

# ── Load model (for model.genes only — no inference) ─────────────────────────
model = None
for ckpt in CKPT_CANDIDATES:
    if not os.path.exists(ckpt):
        continue
    try:
        print(f"Trying checkpoint: {ckpt}")
        model = scPRINT2.load_from_checkpoint(
            ckpt,
            precpt_gene_emb=None,
            gene_pos_file=None,
            weights_only=False,
        )
        model.eval()
        print(f"  loaded OK: {ckpt}")
        break
    except Exception as e:
        print(f"  failed: {e}")

if model is None:
    raise RuntimeError("Could not load any checkpoint from CKPT_CANDIDATES")

print(f"  model.genes: {len(model.genes)} genes")

# ── Per-dataset loop ──────────────────────────────────────────────────────────
for name, url in datasets.items():
    out_path = os.path.join(OUT_DIR, f"subadata_{name}.h5ad")
    if os.path.exists(out_path):
        print(f"\n[{name}] already done, skipping.")
        continue

    print(f"\n{'='*60}\nProcessing: {name}\n{'='*60}")

    try:
        # Load
        h5ad_path = os.path.join(LOC, name + ".h5ad")
        adata = sc.read(h5ad_path, backup_url=url) if url else sc.read(h5ad_path)
        print(f"  loaded: {adata.shape}")

        # Organism tag if missing
        is_symbol = False
        if "organism_ontology_term_id" not in adata.obs.columns:
            adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
            is_symbol = True

        # Preprocessor (exact notebook params)
        preprocessor = Preprocessor(
            force_preprocess=True,
            skip_validate=True,
            is_symbol=is_symbol,
            do_postp=(model.expr_emb_style == "metacell"),
        )
        adata = adata[np.array(adata.X.sum(1)).flatten() > 1000]
        adata = preprocessor(adata)
        print(f"  after preprocess: {adata.shape}")

        # Sub-sample cells (max 5000, like notebook)
        if adata.n_obs > 5000:
            idx = np.random.choice(adata.n_obs, 5000, replace=False)
            adata = adata[idx].copy()

        # Ensure highly_variable exists (without Denoiser it may be missing)
        # Match Denoiser params:
        # sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=self.max_len, span=0.99)
        if "highly_variable" not in adata.var.columns:
            mean_nnz = (
                float((adata.X > 0).sum(1).mean())
                if not sp.issparse(adata.X)
                else float(adata.X.sign().sum(1).mean())
            )
            max_len = 3000 if mean_nnz < 2000 else 5000
            print(
                f"  highly_variable missing -> computing HVG (seurat_v3, n_top_genes={max_len}, span=0.99)"
            )
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=max_len,
                span=0.99,
            )

        # Filter genes: model.genes & highly_variable
        gene_mask = adata.var.index.isin(model.genes) & adata.var[
            "highly_variable"
        ].astype(bool)
        subadata = adata[:, gene_mask].copy()
        print(f"  subadata: {subadata.shape}")

        # Store raw counts, then downsample X
        subadata.layers["true"] = subadata.X.copy()
        subadata.X = downsample_profile(
            torch.Tensor(subadata.layers["true"].toarray()), 0.7
        )

        subadata.write_h5ad(out_path)
        print(f"  saved → {out_path}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"  ERROR on {name}: {e}")

print("\n=== Step 1 complete ===")
