# basic_umap_scgpt.py
# Minimal UMAP plot from obsm["X_scgpt_FT"], colored by a given cell-type column.

import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- user settings ---
INPUT_H5AD = "/iridisfs/ddnb/Luke/scFMs/scGPT/outputs/onek1k_test15donor__ft-onek1k_train951don__ep3__20251006-1152/onek1k_test15donors_with_X_scgpt_FT.h5ad"     # path to AnnData
CELLTYPE_COL = "predicted.celltype.l2"         # obs column to color by
OUTPUT_PNG = "/iridisfs/ddnb/Luke/scFMs/scGPT/outputs/onek1k_test15donor__ft-onek1k_train951don__ep3__20251006-1152/umap.png"   # output image path
# ---------------------

# load
adata = ad.read_h5ad(INPUT_H5AD)

# sanity checks
if "X_scgpt_FT" not in adata.obsm:
    raise KeyError("obsm['X_scgpt_FT'] not found.")
if CELLTYPE_COL not in adata.obs.columns:
    adata.obs[CELLTYPE_COL] = "unknown"

# compute neighbors + umap using the finetuned embeddings
sc.pp.neighbors(adata, use_rep="X_scgpt_FT", n_neighbors=15, metric="cosine")
sc.tl.umap(adata, random_state=0)

# prep dataframe for plotting
umap = adata.obsm["X_umap"]
df = pd.DataFrame(umap, columns=["UMAP1", "UMAP2"])
df["label"] = adata.obs[CELLTYPE_COL].astype(str).values

# color map by category
cats = pd.Categorical(df["label"])
df["label"] = cats
colors = plt.cm.get_cmap("tab20", len(cats.categories)).colors
lut = {cat: colors[i % len(colors)] for i, cat in enumerate(cats.categories)}
point_colors = [lut[c] for c in df["label"]]

# plot
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(df["UMAP1"], df["UMAP2"], s=1.2, c=point_colors, linewidths=0)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("UMAP of finetuned scGPT embeddings")
ax.set_aspect("equal", adjustable="datalim")  # avoid warping

# simple legend (outside)
handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                      markersize=6, color=lut[c], label=str(c))
           for c in cats.categories]
leg = ax.legend(handles=handles, title=CELLTYPE_COL,
                bbox_to_anchor=(1.02, 1), loc="upper left",
                borderaxespad=0., frameon=False, fontsize=8, title_fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUTPUT_PNG}")
