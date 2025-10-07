#!/usr/bin/env python3
"""
Basic UMAP plot from scGPT or finetuned scGPT embeddings.
Automatically uses obsm['X_scgpt_FT'] if present, otherwise falls back to obsm['X_scgpt'].

Usage (called by run_umap.sh):
    python3 plot_umaps.py <input.h5ad> <cell_type_col> [output.png]
"""

import sys
import os
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_umaps.py <input.h5ad> <cell_type_col> [output.png]")
        sys.exit(1)

    input_path = sys.argv[1]
    celltype_col = sys.argv[2]
    output_png = sys.argv[3] if len(sys.argv) > 3 else f"{os.path.splitext(os.path.basename(input_path))[0]}_umap.png"

    print(f"Loading AnnData: {input_path}")
    adata = ad.read_h5ad(input_path)

    # choose embedding automatically
    emb_key = None
    if "X_scgpt_FT" in adata.obsm:
        emb_key = "X_scgpt_FT"
    elif "X_scgpt" in adata.obsm:
        emb_key = "X_scgpt"
    else:
        raise KeyError("Neither obsm['X_scgpt_FT'] nor obsm['X_scgpt'] found in AnnData.")

    print(f"Using embeddings from obsm['{emb_key}'].")

    if celltype_col not in adata.obs.columns:
        print(f"Warning: {celltype_col} not found in obs; using 'unknown'.")
        adata.obs[celltype_col] = "unknown"

    print("Computing neighbours and UMAP ...")
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15, metric="cosine")
    sc.tl.umap(adata)

    umap = adata.obsm["X_umap"]
    df = pd.DataFrame(umap, columns=["UMAP1", "UMAP2"])
    df["Cell Type"] = adata.obs[celltype_col].astype(str).values

    # colour map
    cats = pd.Categorical(df["Cell Type"])
    cmap = plt.cm.get_cmap("tab20", len(cats.categories))
    lut = {cat: cmap(i % cmap.N) for i, cat in enumerate(cats.categories)}
    colors = [lut[c] for c in df["Cell Type"]]

    # plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["UMAP1"], df["UMAP2"], s=1.5, c=colors, linewidths=0)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP of {emb_key} embeddings")
    ax.set_aspect("equal", adjustable="datalim")

    # legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=lut[c], label=c, markersize=6)
        for c in cats.categories
    ]
    ax.legend(handles=handles, title=celltype_col,
              bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False, fontsize=8, title_fontsize=9)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
