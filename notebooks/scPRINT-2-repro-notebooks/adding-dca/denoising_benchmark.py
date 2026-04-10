"""
denoising_benchmark.py
Run scPRINT / MAGIC / DCA denoising benchmark across all datasets.
Adapted from denoising_V3.ipynb — no interactive display, saves all figures.
"""
import os, sys, json, subprocess, tempfile, urllib.request
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from scipy.stats import spearmanr
import pandas as pd
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR   # script lives at ~/scPRINT/
LOC = "/pasteur/appa/scratch/jkalfon/data/spcrint_data/temp/"
LOC2 = os.path.join(PROJECT_ROOT, "models") + "/"
OUT  = os.path.join(PROJECT_ROOT, "notebooks/scPRINT-2-repro-notebooks/denoising_results/")
os.makedirs(LOC2, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

DCA_PYTHON = os.path.join(PROJECT_ROOT, ".dca-env", "bin", "python3")

# ─── Imports ──────────────────────────────────────────────────────────────────
from scprint2 import scPRINT2
from scdataloader import Preprocessor
from scdataloader.utils import load_genes
from scprint2.tasks import Denoiser
from scprint2.model.utils import downsample_profile
from scprint2.tasks.denoise import plot_cell_depth_wise_corr_improvement

torch.set_float32_matmul_precision("medium")

# ─── Load model ───────────────────────────────────────────────────────────────
model_checkpoint_file = "w937u4o1.ckpt"
ckpt_path = "/pasteur/appa/scratch/jkalfon/scprint/small-v2.ckpt"

print("Loading model from", ckpt_path)
model = scPRINT2.load_from_checkpoint(ckpt_path, precpt_gene_emb=None, gene_pos_file=None)
if not torch.cuda.is_available():
    model = model.to(torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")
model = model.to(device)

missing = set(model.genes) - set(load_genes(model.organisms).index)
if missing:
    print(f"Fixing {len(missing)} gene mismatches...")
    model._rm_genes(missing)

# ─── Datasets ─────────────────────────────────────────────────────────────────
datasets = {
    "retina":              "https://datasets.cellxgene.cziscience.com/53bd4177-79c6-40c8-b84d-ff300dcf1b5b.h5ad",
    "kidney":              "https://datasets.cellxgene.cziscience.com/01bc7039-961f-4c24-b407-d535a2a7ba2c.h5ad",
    "pancreas":            "https://figshare.com/ndownloader/files/24539828",
    "intestine":           "https://datasets.cellxgene.cziscience.com/d9a99b4a-3755-47c4-8eb5-09821ffbde17.h5ad",
    "glio_smart_highdepth":"https://datasets.cellxgene.cziscience.com/6ec440b4-542a-4022-ac01-56f812e25593.h5ad",
    "lung_smart":          "https://datasets.cellxgene.cziscience.com/6ebba0e0-a159-406f-8095-451115673a2c.h5ad",
    "SRX24486462":         None,
    "SRX22526970":         None,
}

# ─── Helper: save current figure ──────────────────────────────────────────────
def savefig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ─── Main benchmark loop ──────────────────────────────────────────────────────
res = {}

for name, url in datasets.items():
    print(f"\n{'='*60}\ndoing {name}\n{'='*60}")

    h5ad_path = LOC + name + ".h5ad"
    adata = sc.read(h5ad_path, backup_url=url)
    max_len = 3000 if (adata.X > 0).sum(1).mean() < 2_000 else 5000
    is_symbol = False
    if "organism_ontology_term_id" not in adata.obs.columns:
        adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
        is_symbol = True

    preprocessor = Preprocessor(
        force_preprocess=True,
        skip_validate=True,
        is_symbol=is_symbol,
        do_postp=model.expr_emb_style == "metacell",
    )
    adata = adata[adata.X.sum(1) > 1000]
    adata = preprocessor(adata)
    if model.expr_emb_style == "metacell":
        sc.pp.neighbors(adata, use_rep="X_pca")
    print(f"  mean depth: {adata.X.sum(1).mean():.0f}")
    adata.layers["true"] = adata.X.copy()

    # ── scPRINT ──────────────────────────────────────────────────────────────
    denoise = Denoiser(
        batch_size=40 if (adata.X > 0).sum(1).mean() < 2_000 else 20,
        max_len=max_len,
        max_cells=5_000,
        doplot=False,
        num_workers=8,
        predict_depth_mult=10,
        downsample_expr=0.7,
        additional_info=True,
        apply_zero_pred=True,
    )
    res["scprint_" + name], idx, nadata = denoise(model, adata)
    print("  scPRINT:", res["scprint_" + name])

    # ── MAGIC ────────────────────────────────────────────────────────────────
    subadata = (
        adata[idx, adata.var.index.isin(model.genes) & adata.var.highly_variable].copy()
        if idx is not None
        else adata[:, adata.var.index.isin(model.genes) & adata.var.highly_variable].copy()
    )
    print(f"  subadata shape: {subadata.shape}")
    subadata.X = downsample_profile(torch.Tensor(subadata.X.toarray()), 0.7)
    denoised_adata = sc.external.pp.magic(
        subadata.copy(), copy=True, n_jobs=10, solver="approximate", verbose=True
    )
    reco  = denoised_adata.X
    true  = subadata.layers["true"].toarray()
    noisy = subadata.X if hasattr(subadata.X, "toarray") == False else np.array(subadata.X)

    corr_coef, _ = spearmanr(
        np.vstack([reco[true != 0], noisy[true != 0], true[true != 0]]).T
    )
    res["magic_" + name] = {
        "reco2noisy": corr_coef[0, 1],
        "reco2full":  corr_coef[0, 2],
        "noisy2full": corr_coef[1, 2],
    }

    # subsampled cell-wise
    r, n, t = reco, np.array(noisy), true
    if r.shape[0] >= 3000:
        _idx = np.random.choice(r.shape[0], 3000, replace=False)
        r, n, t = r[_idx], n[_idx], t[_idx]

    cell_wise = np.array([spearmanr(r[i][t[i]!=0], t[i][t[i]!=0])[0] for i in range(r.shape[0])])
    torm      = np.array([spearmanr(n[i][t[i]!=0], t[i][t[i]!=0])[0] for i in range(r.shape[0])])
    cell_wise -= torm

    print(f"  MAGIC cell_wise_to_noisy: {np.mean(cell_wise):.4f}")
    plot_cell_depth_wise_corr_improvement(cell_wise, (t > 0).sum(1))
    savefig(f"magic_depthwise_{name}.png")
    print("  MAGIC:", res["magic_" + name])

    # ── DCA (isolated venv) ──────────────────────────────────────────────────
    if os.path.exists(DCA_PYTHON):
        dca_adata = subadata.copy()
        dca_adata.X = downsample_profile(torch.Tensor(dca_adata.layers["true"].toarray()), 0.7)
        _dca_in  = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
        _dca_out = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False).name
        dca_adata.write_h5ad(_dca_in)

        _dca_code = f"""
import scanpy as sc, anndata as ad
adata = ad.read_h5ad("{_dca_in}")
result = sc.external.pp.dca(adata, copy=True, verbose=False)
result.write_h5ad("{_dca_out}")
print("DCA done, shape:", result.X.shape)
"""
        print("  Running DCA (subprocess)...")
        _proc = subprocess.run([DCA_PYTHON, "-c", _dca_code],
                               capture_output=True, text=True, timeout=600)
        if _proc.returncode != 0:
            print("  DCA stderr:", _proc.stderr[-800:])
            res["dca_" + name] = {"reco2noisy": None, "reco2full": None, "noisy2full": None}
        else:
            if _proc.stdout: print(_proc.stdout[-300:])
            _dca_result  = ad.read_h5ad(_dca_out)
            reco_dca  = np.array(_dca_result.X)
            true_dca  = subadata.layers["true"].toarray()
            noisy_dca = np.array(dca_adata.X)

            corr_dca, _ = spearmanr(
                np.vstack([reco_dca[true_dca!=0], noisy_dca[true_dca!=0], true_dca[true_dca!=0]]).T
            )
            res["dca_" + name] = {
                "reco2noisy": corr_dca[0,1],
                "reco2full":  corr_dca[0,2],
                "noisy2full": corr_dca[1,2],
            }
            r2, n2, t2 = reco_dca, noisy_dca, true_dca
            if r2.shape[0] >= 3000:
                _i = np.random.choice(r2.shape[0], 3000, replace=False)
                r2, n2, t2 = r2[_i], n2[_i], t2[_i]
            cw_dca  = np.array([spearmanr(r2[i][t2[i]!=0], t2[i][t2[i]!=0])[0] for i in range(r2.shape[0])])
            torm_dca= np.array([spearmanr(n2[i][t2[i]!=0], t2[i][t2[i]!=0])[0] for i in range(r2.shape[0])])
            cw_dca -= torm_dca
            print(f"  DCA cell_wise_to_noisy: {np.mean(cw_dca):.4f}")
            plot_cell_depth_wise_corr_improvement(cw_dca, (t2 > 0).sum(1))
            savefig(f"dca_depthwise_{name}.png")
            print("  DCA:", res["dca_" + name])

        for f in (_dca_in, _dca_out):
            try: os.unlink(f)
            except: pass
    else:
        print(f"  DCA venv not found at {DCA_PYTHON}, skipping DCA")
        res["dca_" + name] = {"reco2noisy": None, "reco2full": None, "noisy2full": None}

    print(f"  Done {name}.")

# ─── Save results JSON ────────────────────────────────────────────────────────
results_path = os.path.join(OUT, "benchmark_results.json")
with open(results_path, "w") as f:
    json.dump(res, f, indent=2)
print(f"\nResults saved to {results_path}")
print(json.dumps(res, indent=2))

# ─── Summary plot ────────────────────────────────────────────────────────────
depths = {
    "retina":3800, "kidney":5300, "pancreas":13000,
    "intestine":21693, "glio_smart_highdepth":38469,
    "lung_smart":160000, "SRX24486462":400000, "SRX22526970":1000000,
}
quality = {
    "retina":"low","kidney":"low","pancreas":"mid",
    "intestine":"mid","glio_smart_highdepth":"mid",
    "lung_smart":"high","SRX24486462":"high","SRX22526970":"high",
}
rows = []
for name in datasets:
    q = quality.get(name, "mid")
    for method in ["scprint", "magic", "dca"]:
        key = f"{method}_{name}"
        if key in res and isinstance(res[key], dict):
            val = res[key].get("reco2full")
            if val is not None:
                rows.append({"quality": q, "name": method, "value": val * 100})

if rows:
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="quality", y="value", hue="name", order=["low","mid","high"])
    plt.title("Magic vs DCA vs scPRINT denoising per dataset quality (depth)")
    plt.ylabel("Spearman reco2full (%)")
    savefig("denoising_comparison_boxplot.png")
    print("Summary boxplot saved.")
else:
    print("No results to plot yet.")

print("\nAll done!")
