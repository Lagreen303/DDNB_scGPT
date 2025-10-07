#!/usr/bin/env python3
"""
Compute AvgBIO scores (ASW, ARI, NMI) for an AnnData embedding.

Usage:
    python3 compute_avgbio.py --input <file.h5ad> --embedding-key <key> --ground-truth-col <col>
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def flush_print(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AvgBIO (ASW, ARI, NMI) scores from embeddings in an AnnData file."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input .h5ad file."
    )
    parser.add_argument(
        "--embedding-key", required=True,
        help="Key in adata.obsm containing embeddings (e.g. X_scgpt_FT or X_scgpt)."
    )
    parser.add_argument(
        "--ground-truth-col", required=True,
        help="Column in adata.obs with ground-truth labels for clustering comparison."
    )
    args = parser.parse_args()

    input_file = args.input
    embedding_key = args.embedding_key
    ground_truth_col = args.ground_truth_col

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(os.path.dirname(input_file), base_name)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "avgbio.csv")

    # === Load data ===
    flush_print(f"Loading AnnData from {input_file} ...")
    adata = ad.read_h5ad(input_file)

    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found in adata.obsm.")
    if ground_truth_col not in adata.obs.columns:
        raise KeyError(f"Ground truth column '{ground_truth_col}' not found in adata.obs.")

    embedding = adata.obsm[embedding_key]
    labels = adata.obs[ground_truth_col].astype(str).values

    # === Louvain clustering ===
    flush_print("Running Louvain clustering across resolutions ...")
    adata.obsm["X_temp"] = embedding
    sc.pp.neighbors(adata, use_rep="X_temp", n_neighbors=15)

    best_nmi, best_res, best_labels = -1, 0.0, None
    for res in np.linspace(0.1, 2.0, 20):
        sc.tl.louvain(adata, resolution=res, key_added=f"louvain_{res:.2f}")
        cluster_labels = adata.obs[f"louvain_{res:.2f}"].values
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        if nmi > best_nmi:
            best_nmi, best_res, best_labels = nmi, res, cluster_labels

    flush_print(f"Best Louvain resolution: {best_res:.2f} (NMI = {best_nmi:.3f})")
    adata.obs["best_louvain"] = best_labels

    # === Compute metrics ===
    flush_print("Computing metrics ...")
    asw = silhouette_score(embedding, labels)
    asw_norm = (asw + 1) / 2
    ari = adjusted_rand_score(labels, best_labels)
    nmi = normalized_mutual_info_score(labels, best_labels)
    avg_bio = (asw_norm + ari + nmi) / 3

    flush_print("\nEvaluation Metrics:")
    flush_print(f"ASW        : {asw:.3f} (normalized: {asw_norm:.3f})")
    flush_print(f"ARI        : {ari:.3f}")
    flush_print(f"NMI        : {nmi:.3f}")
    flush_print(f"AvgBIO     : {avg_bio:.3f}")
    flush_print(f"Best Res.  : {best_res:.2f}")

    # === Save results ===
    results = pd.DataFrame({
        "ASW": [asw],
        "ASW_norm": [asw_norm],
        "ARI": [ari],
        "NMI": [nmi],
        "AvgBIO": [avg_bio],
        "Best_Louvain_Resolution": [best_res],
    })
    results.to_csv(output_csv, index=False)
    flush_print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    main()
