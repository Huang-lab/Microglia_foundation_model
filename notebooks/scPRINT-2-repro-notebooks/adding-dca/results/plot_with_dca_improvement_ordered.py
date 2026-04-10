import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

res_path = "/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/dca_results_v2.json"
with open(res_path) as f:
    res = json.load(f)

datasets = [
    "retina", "kidney", "pancreas", "intestine",
    "glio_smart_highdepth", "lung_smart", "SRX24486462", "SRX22526970"
]

dca_impr = []
for d in datasets:
    m = res[f"dca_{d}"]
    dca_impr.append((m["reco2full"] - m["noisy2full"]) * 100)

data = {
    "depth": [3_800, 5_300, 13_000, 21_693, 38_469, 160_000, 400_000, 1_000_000],
    "magic": [29.0, 25.0, 31.7, 49.2, 37.0, 20.5, 12.5, 25.2],
    "dca": dca_impr,
    "scprint": [17.3, 16.2, 23.6, 41.7, 32.3, 24.2, 34.2, 39.2],
    "scprint_2": [28.8, 28.4, 39.4, 51.3, 45.0, 36.2, 30.7, 32.8],
    "quality": ["low", "low", "mid", "mid", "mid", "high", "high", "high"],
}

df = pd.DataFrame(data)

order = ["magic", "dca", "scprint", "scprint_2"]
name_map = {
    "magic": "MAGIC",
    "dca": "DCA",
    "scprint": "scPRINT",
    "scprint_2": "scPRINT-2",
}

long = df.melt(
    id_vars=["quality"],
    value_vars=order,
    var_name="name",
    value_name="value",
)
long["name"] = long["name"].map(name_map)
hue_order = ["MAGIC", "DCA", "scPRINT", "scPRINT-2"]

sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(10, 6))
sns.boxplot(data=long, x="quality", y="value", hue="name", hue_order=hue_order)
plt.title("Denoising improvement by dataset quality")
plt.xlabel("Dataset quality bucket")
plt.ylabel("Denoising improvement (%) = (reco2full - noisy2full) * 100")
plt.tight_layout()
out = "/pasteur/appa/homes/jkalfon/scPRINT/notebooks/scPRINT-2-repro-notebooks/denoising_results/denoising_boxplot_dca_v2_improvement_ordered.png"
plt.savefig(out, dpi=220)
print(out)
