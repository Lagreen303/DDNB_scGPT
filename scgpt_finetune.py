from helical.models.scgpt import scGPTFineTuningModel, scGPTConfig
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import anndata as ad
import sys
import pandas as pd
import re
import gzip
import numpy as np
import umap
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scanpy as sc
import torch
import argparse
from sklearn.model_selection import train_test_split

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# GTF Mapping function
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

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def main():
    parser = argparse.ArgumentParser(description='scGPT Finetuning → save test embeddings and UMAP')
    # Order: train, test, cell-type-col, epochs, output, batch-size, gtf, vocab
    parser.add_argument('--train-input', required=True, help='Training data h5ad file path')
    parser.add_argument('--test-input', required=True, help='Test data h5ad file path')
    parser.add_argument('--cell-type-col', default='cell_type', help='Cell type column name (for training labels)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--output', required=True, help='Parent output directory; run subdir will be created inside')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--gtf', required=True, help='GTF file path (gz)')
    parser.add_argument('--vocab', required=True, help='scGPT vocabulary file path (one gene per line)')
    args = parser.parse_args()
    EMB_KEY = 'X_scgpt_FT'
    
    # === Load scGPT config ===
    flush_print("Loading scGPT configuration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scgpt_config = scGPTConfig(batch_size=args.batch_size, device=device)

    # === Load data ===
    flush_print("Loading training data...")
    train_file = args.train_input
    train_data = ad.read_h5ad(train_file)
    train_base_name = os.path.basename(train_file).replace(".h5ad", "")
    
    flush_print("Loading test data...")
    test_file = args.test_input
    test_data = ad.read_h5ad(test_file)
    test_base_name = os.path.basename(test_file).replace(".h5ad", "")
    
    # Create timestamped run directory
    def slug(s: str, maxlen: int = 18) -> str:
        s = re.sub(r'[^a-zA-Z0-9._-]+', '-', s).strip('-').lower()
        return s[:maxlen]

    train_slug = slug(train_base_name)
    test_slug = slug(test_base_name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{test_slug}__ft-{train_slug}__ep{args.epochs}__{stamp}"
    output_dir = os.path.join(args.output, run_name)
    os.makedirs(output_dir, exist_ok=True)
    flush_print(f"[run dir] {output_dir}")

    # Ensure raw counts are used
    if train_data.raw is not None:
        train_data.X = train_data.raw.X
    if test_data.raw is not None:
        test_data.X = test_data.raw.X

    # Filter cells with missing labels (training only)
    flush_print(f"Filtering training cells with missing '{args.cell_type_col}' labels...")
    train_data = train_data[
        train_data.obs[args.cell_type_col].notnull() &
        (train_data.obs[args.cell_type_col] != "nan")
    ]
    train_data.obs[args.cell_type_col] = train_data.obs[args.cell_type_col].astype(str)
    
    # Test can keep labels or not; used only for colouring
    test_data.obs[args.cell_type_col] = test_data.obs.get(args.cell_type_col, pd.Series(index=test_data.obs_names, dtype=str)).astype(str)

    # === GTF gene ID → symbol mapping ===
    flush_print("Parsing GTF...")
    mapping_dict = parse_gtf(args.gtf)

    # Map genes to symbols and de-duplicate (train)
    flush_print("Processing training data (map Ensembl→symbol, dedupe)...")
    train_data.var['ensembl_clean'] = train_data.var_names.str.replace(r'\..*$', '', regex=True)
    train_data.var['gene_symbol'] = train_data.var['ensembl_clean'].map(mapping_dict)
    train_data = train_data[:, train_data.var['gene_symbol'].notnull()]
    train_data = train_data[:, ~train_data.var['gene_symbol'].duplicated()]
    train_data.var_names = train_data.var['gene_symbol']

    # Same for test
    flush_print("Processing test data (map Ensembl→symbol, dedupe)...")
    test_data.var['ensembl_clean'] = test_data.var_names.str.replace(r'\..*$', '', regex=True)
    test_data.var['gene_symbol'] = test_data.var['ensembl_clean'].map(mapping_dict)
    test_data = test_data[:, test_data.var['gene_symbol'].notnull()]
    test_data = test_data[:, ~test_data.var['gene_symbol'].duplicated()]
    test_data.var_names = test_data.var['gene_symbol']

    # === Load scGPT vocabulary and subset to a COMMON, ORDERED gene set (POINT 2) ===
    flush_print("Loading scGPT vocabulary...")
    with open(args.vocab) as f:
        vocab_list = [line.strip() for line in f if line.strip()]

    # Enforce identical features & order across train and test using the vocab order
    common = [g for g in vocab_list if g in train_data.var_names and g in test_data.var_names]
    if len(common) == 0:
        raise RuntimeError("No overlapping genes between train/test after GTF mapping and vocab filter.")
    train_data = train_data[:, common]
    test_data  = test_data[:, common]
    flush_print(f"Using {len(common)} common genes (ordered by vocab).")

    # === Labels for training (deterministic mapping) ===
    flush_print("Preparing training labels...")
    train_cell_types = list(train_data.obs[args.cell_type_col].astype(str).values)
    classes = sorted(set(train_cell_types))
    class_id_dict = {cls: i for i, cls in enumerate(classes)}
    train_labels = [class_id_dict[cell_type] for cell_type in train_cell_types]
    flush_print(f"Classes ({len(classes)}): {classes}")

    flush_print(f"Training samples: {train_data.n_obs}")
    flush_print(f"Test samples: {test_data.n_obs}")

    # === Build and train the finetuning model (classification head for supervision) ===
    flush_print("Creating fine-tuning model...")
    scgpt_fine_tune = scGPTFineTuningModel(
        scGPT_config=scgpt_config,
        fine_tuning_head="classification",
        output_size=len(classes)
    )

    flush_print("Processing training data for model...")
    train_dataset = scgpt_fine_tune.process_data(train_data)

    flush_print(f"Starting fine-tuning for {args.epochs} epochs...")
    scgpt_fine_tune.train(
        train_input_data=train_dataset,
        train_labels=train_labels,
        epochs=int(args.epochs)
    )
    flush_print("Fine-tuning completed!")

    # === Get test embeddings from the finetuned model and save to .obsm ===
    flush_print("Processing test data and extracting embeddings...")
    test_dataset = scgpt_fine_tune.process_data(test_data)
    embeddings = scgpt_fine_tune.get_embeddings(test_dataset)

    # Normalise to (n_cells, d)
    if isinstance(embeddings, (list, tuple)) and hasattr(embeddings[0], "__len__"):
        embeddings = np.vstack([to_numpy(e) for e in embeddings])
    else:
        embeddings = to_numpy(embeddings)

    if embeddings.ndim != 2 or embeddings.shape[0] != test_data.n_obs:
        raise RuntimeError(f"Unexpected embeddings shape: {embeddings.shape} (expected ({test_data.n_obs}, d)).")

    # Save embeddings into AnnData
    test_data.obsm[EMB_KEY] = embeddings
    flush_print(f"Saved test embeddings to obsm['{EMB_KEY}'] with shape {embeddings.shape}")

    # === Neighbours + UMAP using the finetuned embeddings (match plot_umaps) ===
    flush_print("Computing neighbours/UMAP using finetuned embeddings...")
    sc.pp.neighbors(test_data, use_rep=EMB_KEY, n_neighbors=15, metric="cosine")
    sc.tl.umap(test_data, random_state=0)

    # Plot UMAP with plot_umaps style (categorical colors, external legend)
    umap = test_data.obsm["X_umap"]
    df = pd.DataFrame(umap, columns=["UMAP1", "UMAP2"])
    labels = test_data.obs.get(args.cell_type_col, pd.Series(["unknown"] * test_data.n_obs, index=test_data.obs_names)).astype(str)
    df["label"] = pd.Categorical(labels.values)

    colors = plt.cm.get_cmap("tab20", len(df["label"].cat.categories)).colors
    lut = {cat: colors[i % len(colors)] for i, cat in enumerate(df["label"].cat.categories)}
    point_colors = [lut[c] for c in df["label"]]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["UMAP1"], df["UMAP2"], s=1.2, c=point_colors, linewidths=0)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP of finetuned scGPT embeddings")
    ax.set_aspect("equal", adjustable="datalim")

    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=6, color=lut[c], label=str(c))
               for c in df["label"].cat.categories]
    ax.legend(handles=handles, title=args.cell_type_col,
              bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0., frameon=False, fontsize=8, title_fontsize=9)

    fig.tight_layout()
    umap_plot_path = os.path.join(output_dir, f"{test_base_name}_{EMB_KEY}_umap.png")
    fig.savefig(umap_plot_path, dpi=300, bbox_inches='tight')
    flush_print(f"UMAP plot saved to {umap_plot_path}")
    plt.close(fig)

    # === Save artefacts ===
    flush_print("Saving outputs...")
    test_output_file = os.path.join(output_dir, f"{test_base_name}_with_{EMB_KEY}.h5ad")
    test_data.write_h5ad(test_output_file)
    flush_print(f"Test AnnData (with embeddings) saved to {test_output_file}")

    # Save classes used for training (useful for reproducibility)
    import json
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        f.write("\n".join(classes))
    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_id_dict, f, indent=2)
    flush_print("Saved classes.txt and class_mapping.json")

    print(f"Train shape: {train_data.shape} | Test shape: {test_data.shape}")
    print("Done.")

if __name__ == "__main__":
    main()
