from helical.models.scgpt import scGPT, scGPTConfig
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import anndata as ad
import sys
import pandas as pd
import re
import gzip
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scanpy as sc
import torch
import argparse
from datetime import datetime

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

#GTF Mapping function
def parse_gtf(gtf_file):
    ensembl_to_symbol = {}
    with gzip.open(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if fields[2] != 'gene':
                continue
            attributes = fields[8]
            gene_id_match = re.search('gene_id "([^"]+)"', attributes)
            gene_name_match = re.search('gene_name "([^"]+)"', attributes)
            if gene_id_match and gene_name_match:
                ensembl_id = gene_id_match.group(1).split('.')[0]  # Strip version
                gene_symbol = gene_name_match.group(1)
                ensembl_to_symbol[ensembl_id] = gene_symbol
    return ensembl_to_symbol

def main():
    parser = argparse.ArgumentParser(description='scGPT Zeroshot Analysis')
    parser.add_argument('--input', required=True, help='Input h5ad file path')
    parser.add_argument('--output', required=True, help='Parent output directory; run subdir will be created inside')
    parser.add_argument('--gtf', required=True, help='GTF file path')
    parser.add_argument('--vocab', required=True, help='scGPT vocabulary file path')
    parser.add_argument('--cell-type-col', default='cell_type', help='Cell type column name')
    
    args = parser.parse_args()
    
    # === Load scGPT model ===
    flush_print("Loading scGPT configuration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scgpt_config = scGPTConfig(batch_size=8, device=device)
    scgpt = scGPT(configurer=scgpt_config)

    # === Load data ===
    flush_print("Loading data...")
    input_file = args.input
    ann_data = ad.read_h5ad(input_file)
    base_name = os.path.basename(input_file).replace(".h5ad", "")
    # Create timestamped run directory under provided output parent
    def slug(s: str, maxlen: int = 18) -> str:
        s = re.sub(r'[^a-zA-Z0-9._-]+', '-', s).strip('-').lower()
        return s[:maxlen]
    base_slug = slug(base_name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{base_slug}__zs__{stamp}"
    output_dir = os.path.join(args.output, run_name)
    os.makedirs(output_dir, exist_ok=True)
    flush_print(f"[run dir] {output_dir}")

    # Ensure raw counts are used
    if ann_data.raw is not None:
        ann_data.X = ann_data.raw.X

    flush_print(f"Filtering cells with missing '{args.cell_type_col}' labels...")
    ann_data = ann_data[
        ann_data.obs[args.cell_type_col].notnull() &
        (ann_data.obs[args.cell_type_col] != "nan")
    ]
    ann_data.obs[args.cell_type_col] = ann_data.obs[args.cell_type_col].astype(str)

    flush_print("Parsing GTF...")
    mapping_dict = parse_gtf(args.gtf)

    # Map Ensembl IDs to gene symbols safely inside AnnData.var
    flush_print("Mapping Ensembl IDs to gene symbols...")
    ann_data.var['ensembl_clean'] = ann_data.var_names.str.replace(r'\..*$', '', regex=True)
    ann_data.var['gene_symbol'] = ann_data.var['ensembl_clean'].map(mapping_dict)
    # Remove genes without gene symbol mapping
    ann_data = ann_data[:, ann_data.var['gene_symbol'].notnull()]
    # Remove duplicate gene symbols
    ann_data = ann_data[:, ~ann_data.var['gene_symbol'].duplicated()]
    # Assign gene symbols to var_names
    ann_data.var_names = ann_data.var['gene_symbol']

    # === Load scGPT vocabulary and subset ===
    flush_print("Loading scGPT vocabulary...")
    with open(args.vocab) as f:
        vocab_list = [line.strip() for line in f if line.strip()]

    # Subset AnnData to genes present in vocabulary
    genes_in_vocab = [g for g in ann_data.var_names if g in vocab_list]
    ann_data = ann_data[:, genes_in_vocab]
    # Reorder genes to match vocab order
    ordered_genes = [g for g in vocab_list if g in ann_data.var_names]
    ann_data = ann_data[:, ordered_genes]

    flush_print("Processing data for scGPT...")
    dataset = scgpt.process_data(ann_data)
    flush_print("Computing embeddings...")
    embeddings = scgpt.get_embeddings(dataset)

    # Save embeddings into AnnData
    if isinstance(embeddings, torch.Tensor):
        ann_data.obsm["X_scgpt"] = embeddings.cpu().numpy() if torch.cuda.is_available() else embeddings.numpy()
    else:
        ann_data.obsm["X_scgpt"] = embeddings 

    # Compute neighbors using scGPT embeddings
    sc.pp.neighbors(ann_data, use_rep="X_scgpt")
    # Compute UMAP based on those neighbors
    sc.tl.umap(ann_data)
    ann_data.obsm["X_umap_scgpt"] = ann_data.obsm["X_umap"]

    # Extract UMAP coordinates into a dataframe and plot with consistent styling
    plot_df = pd.DataFrame(ann_data.obsm["X_umap_scgpt"], columns=["UMAP1", "UMAP2"])
    plot_df["Cell Type"] = ann_data.obs[args.cell_type_col].astype(str).to_numpy()

    # Global styling to match other scripts
    plt.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 36,
        'axes.labelsize': 28,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 20,
        'figure.titlesize': 40
    })

    unique_celltypes = sorted(np.unique(plot_df["Cell Type"]))
    palette = sns.color_palette("husl", len(unique_celltypes))
    lut = dict(zip(unique_celltypes, palette))

    flush_print("Plotting and saving UMAP...")
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=plot_df, x="UMAP1", y="UMAP2", hue="Cell Type", palette=lut, s=1.2)
    plt.title("UMAP: Colored by Cell Type", fontsize=50, fontweight='bold')
    plt.xlabel("UMAP1", fontsize=28)
    plt.ylabel("UMAP2", fontsize=28)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24, title_fontsize=28, title="Cell Type")
    plt.tight_layout()
    umap_plot_path = os.path.join(output_dir, f"{base_name}_scGPT_umap.png")
    plt.savefig(umap_plot_path, dpi=300, bbox_inches='tight')
    flush_print(f"UMAP plot saved to {umap_plot_path}")
    plt.close()

    #save the AnnData object with embeddings
    flush_print("Saving AnnData with scGPT embeddings...")
    output_file = os.path.join(output_dir, f"{base_name}_scGPT.h5ad")
    ann_data.write_h5ad(output_file)
    flush_print(f"Saving AnnData to {output_file}...")

    print(f"AnnData shape after vocab filtering: {ann_data.shape}")
    print("Embedding variance:", np.var(embeddings, axis=0).mean())

if __name__ == "__main__":
    main()