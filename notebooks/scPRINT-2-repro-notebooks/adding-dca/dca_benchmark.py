"""
dca_benchmark.py
Run DCA only on all denoising benchmark datasets.
Uses tools/dca submodule (jkobject/dca fork, keras-patched) with .dca-env Python.
Called via subprocess from the main scprint_test env so we can use scdataloader/scprint2
for preprocessing, then hand off to DCA env.
Results saved to denoising_results/dca_results.json
"""
import os, sys, json, subprocess, tempfile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import scanpy as sc
from scipy.stats import spearmanr

from scdataloader import Preprocessor
from scprint2.model.utils import downsample_profile

PROJECT_ROOT = "/pasteur/appa/homes/jkalfon/scPRINT"
LOC  = "/pasteur/appa/scratch/jkalfon/data/spcrint_data/temp/"
OUT  = os.path.join(PROJECT_ROOT, "notebooks/scPRINT-2-repro-notebooks/denoising_results/")
DCA_PYTHON = os.path.join(PROJECT_ROOT, ".dca-env/bin/python3")
DCA_SRC    = os.path.join(PROJECT_ROOT, "tools/dca")
os.makedirs(OUT, exist_ok=True)

datasets = {
    "retina":               "https://datasets.cellxgene.cziscience.com/53bd4177-79c6-40c8-b84d-ff300dcf1b5b.h5ad",
    "kidney":               "https://datasets.cellxgene.cziscience.com/01bc7039-961f-4c24-b407-d535a2a7ba2c.h5ad",
    "pancreas":             "https://figshare.com/ndownloader/files/24539828",
    "intestine":            "https://datasets.cellxgene.cziscience.com/d9a99b4a-3755-47c4-8eb5-09821ffbde17.h5ad",
    "glio_smart_highdepth": "https://datasets.cellxgene.cziscience.com/6ec440b4-542a-4022-ac01-56f812e25593.h5ad",
    "lung_smart":           "https://datasets.cellxgene.cziscience.com/6ebba0e0-a159-406f-8095-451115673a2c.h5ad",
    "SRX24486462":          None,
    "SRX22526970":          None,
}

res = {}

for name, url in datasets.items():
    print(f"\n{'='*60}\ndoing {name}\n{'='*60}", flush=True)

    adata = sc.read(LOC + name + ".h5ad", backup_url=url)
    is_symbol = "organism_ontology_term_id" not in adata.obs.columns
    if is_symbol:
        adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"

    preprocessor = Preprocessor(force_preprocess=True, skip_validate=True,
                                 is_symbol=is_symbol, do_postp=False)
    adata = adata[adata.X.sum(1) > 1000]
    adata = preprocessor(adata)
    print(f"  shape: {adata.shape}, mean depth: {adata.X.sum(1).mean():.0f}", flush=True)
    adata.layers["true"] = adata.X.copy()

    if adata.shape[0] > 5000:
        idx = np.random.choice(adata.shape[0], 5000, replace=False)
        subadata = adata[idx].copy()
    else:
        subadata = adata.copy()

    subadata.X = downsample_profile(torch.Tensor(subadata.X.toarray()), 0.7).numpy()

    # filter all-zero genes required by DCA
    dca_input = subadata.copy()
    sc.pp.filter_genes(dca_input, min_cells=1)
    print(f"  after gene filter: {dca_input.shape}", flush=True)

    # write to temp file for DCA subprocess
    _dca_in  = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
    _dca_out = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
    dca_input.write_h5ad(_dca_in)

    _dca_code = f"""
import sys
sys.path.insert(0, "{DCA_SRC}")
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import anndata as ad
from dca.api import dca
adata = ad.read_h5ad("{_dca_in}")
result = dca(adata, return_info=False, copy=True, verbose=False)
result.write_h5ad("{_dca_out}")
print("DCA done, shape:", result.X.shape)
"""
    print("  Running DCA...", flush=True)
    _proc = subprocess.run([DCA_PYTHON, "-c", _dca_code],
                           capture_output=True, text=True, timeout=600)

    if _proc.returncode != 0:
        print(f"  DCA FAILED:\n{_proc.stderr[-1000:]}", flush=True)
        res["dca_" + name] = {"reco2noisy": None, "reco2full": None, "noisy2full": None}
    else:
        if _proc.stdout: print(_proc.stdout[-200:], flush=True)
        import anndata as _ad
        result    = _ad.read_h5ad(_dca_out)
        reco_dca  = np.array(result.X)
        true_dca  = dca_input.layers["true"].toarray()
        noisy_dca = np.array(dca_input.X)

        corr, _ = spearmanr(
            np.vstack([reco_dca[true_dca!=0], noisy_dca[true_dca!=0], true_dca[true_dca!=0]]).T
        )
        res["dca_" + name] = {
            "reco2noisy": corr[0,1], "reco2full": corr[0,2], "noisy2full": corr[1,2]
        }
        r, n, t = reco_dca, noisy_dca, true_dca
        if r.shape[0] >= 3000:
            _i = np.random.choice(r.shape[0], 3000, replace=False)
            r, n, t = r[_i], n[_i], t[_i]
        cw   = np.array([spearmanr(r[i][t[i]!=0], t[i][t[i]!=0])[0] for i in range(r.shape[0])])
        torm = np.array([spearmanr(n[i][t[i]!=0], t[i][t[i]!=0])[0] for i in range(r.shape[0])])
        print(f"  DCA cell_wise_to_noisy: {np.mean(cw-torm):.4f}", flush=True)
        print(f"  DCA: {res['dca_' + name]}", flush=True)

    for f in (_dca_in, _dca_out):
        try: os.unlink(f)
        except: pass

results_path = os.path.join(OUT, "dca_results.json")
with open(results_path, "w") as f:
    json.dump(res, f, indent=2)
print(f"\nResults saved to {results_path}")
print(json.dumps(res, indent=2))
print("\nAll done!")
