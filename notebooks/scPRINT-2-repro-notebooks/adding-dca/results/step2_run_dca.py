"""
Step 2 (DCA env): reads subadata saved by step1, runs DCA, computes metrics.
Each subadata has:
  - X          = downsampled (noisy) raw counts
  - layers["true"] = full raw counts
Same cells & genes as scPRINT/Magic in the notebook.
"""
import sys
sys.path.insert(0, '/pasteur/appa/homes/jkalfon/scPRINT/tools/dca')

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os, json
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy.stats import spearmanr
from dca.api import dca

# ── Config ────────────────────────────────────────────────────────────────────
SUBADATA_DIR = "/pasteur/appa/scratch/jkalfon/data/spcrint_data/dca_subadata/"
OUT_JSON     = "/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_results_v2.json"

datasets = [
    "retina", "kidney", "pancreas", "intestine",
    "glio_smart_highdepth", "lung_smart", "SRX24486462", "SRX22526970"
]

results = {}

for name in datasets:
    subadata_path = os.path.join(SUBADATA_DIR, f"subadata_{name}.h5ad")
    print(f"\n{'='*60}")
    print(f"DCA on: {name}")
    print(f"{'='*60}")

    if not os.path.exists(subadata_path):
        print(f"  MISSING: {subadata_path}")
        results[f"dca_{name}"] = {"reco2noisy": None, "reco2full": None, "noisy2full": None}
        continue

    try:
        subadata = ad.read_h5ad(subadata_path)
        print(f"  shape: {subadata.shape}")

        true_arr  = subadata.layers["true"].toarray() if sp.issparse(subadata.layers["true"]) \
                    else np.array(subadata.layers["true"])
        noisy_arr = subadata.X.toarray() if sp.issparse(subadata.X) \
                    else np.array(subadata.X)

        print(f"  X (noisy) dtype: {noisy_arr.dtype}, sample: {noisy_arr[0,:5]}")
        print(f"  true dtype: {true_arr.dtype}, sample: {true_arr[0,:5]}")

        # DCA needs integer-like counts in X and no all-zero genes
        dca_adata = subadata.copy()
        dca_adata.X = sp.csr_matrix(np.round(noisy_arr).astype(np.float32))

        # Drop all-zero genes (on noisy counts) to satisfy DCA assertion
        xsum = np.array(dca_adata.X.sum(0)).ravel()
        keep = xsum > 0
        dca_adata = dca_adata[:, keep].copy()

        result = dca(dca_adata, copy=True, verbose=False)
        reco_arr = np.array(result.X)
        print(f"  DCA done, shape: {reco_arr.shape}")

        # Align true/noisy arrays to kept genes
        true_kept = true_arr[:, keep]
        noisy_kept = noisy_arr[:, keep]

        # Metrics (cell-wise, non-zero only)
        mask = true_kept != 0
        corr_coef, _ = spearmanr(
            np.vstack([reco_arr[mask], noisy_kept[mask], true_kept[mask]]).T
        )
        metrics = {
            "reco2noisy": float(corr_coef[0, 1]),
            "reco2full":  float(corr_coef[0, 2]),
            "noisy2full": float(corr_coef[1, 2]),
        }
        print(f"  metrics: {metrics}")
        results[f"dca_{name}"] = metrics

    except Exception as e:
        import traceback
        traceback.print_exc()
        results[f"dca_{name}"] = {"reco2noisy": None, "reco2full": None, "noisy2full": None}
        print(f"  FAILED: {e}")

# Save
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== Results saved to {OUT_JSON} ===")
print(json.dumps(results, indent=2))
