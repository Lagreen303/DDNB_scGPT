# DDNB scGPT wrapper

Opinionated wrapper around Helical's scGPT workflows, designed to be run via Slurm-ready bash scripts. You pass parameters to the scripts; they handle the container invocation and outputs.

Helical docs: [Helical on GitHub](https://github.com/helicalAI/helical)

## Contents

- `run_zeroshot.sh` — Zeroshot embeddings + UMAP from a single `.h5ad`
- `run_finetune.sh` — Finetune scGPT on train `.h5ad`, embed test `.h5ad`, plot UMAP
- `scgpt_zeroshot.py`, `scgpt_finetune.py` — Internal Python drivers used by the scripts
- `get_scgpt_vocab.py` — Extract `scgpt_vocab.txt` from a cached `vocab.json`

## Requirements

- Apptainer/Singularity with a Helical container available (scripts call Apptainer)
- GTF annotation file (e.g., `gencode.v48.basic.annotation.gtf.gz`)
- scGPT vocabulary file (`scgpt_vocab.txt`)
- Input data in `.h5ad` format (AnnData) with a cell-type column

### Container (Apptainer/Singularity)

If needed, build from a `singularity.def` and shell in:

```bash
apptainer build --sandbox singularity/helical singularity.def
apptainer shell --nv --fakeroot singularity/helical/
```

Note: The provided scripts already execute inside an Apptainer environment and expect a container at `/iridisfs/ddnb/Luke/helical_sb` with necessary binds. Adjust paths if your environment differs.

## Vocabulary generation

If you don't have `scgpt_vocab.txt`, generate it from the cached `vocab.json`:

```bash
python3 get_scgpt_vocab.py
```

This reads `.cache/helical/models/scgpt/scGPT_CP/vocab.json` and writes `scgpt_vocab.txt` in this directory (one gene symbol per line).

## How to run (use the bash scripts)

### Zeroshot

Script: `run_zeroshot.sh`

Usage:

```bash
./run_zeroshot.sh <input.h5ad> <output_dir> [gtf_file] [vocab_file] [cell_type_col]
```

Parameters:
- `input.h5ad` (required): Single-cell dataset
- `output_dir` (required): Parent directory; a timestamped run subdirectory is created
- `gtf_file` (default: `gencode.v48.basic.annotation.gtf.gz`): Gene annotation (gz)
- `vocab_file` (default: `scgpt_vocab.txt`): One gene symbol per line
- `cell_type_col` (default: `cell_type`): Obs column used for colouring

Example:

```bash
./run_zeroshot.sh data.h5ad results/
```

Outputs inside run dir:
- `<basename>_scGPT.h5ad` with `.obsm['X_scgpt']`
- `<basename>_scGPT_umap.png`

### Finetune

Script: `run_finetune.sh`

Usage:

```bash
./run_finetune.sh <train.h5ad> <test.h5ad> <cell_type_col> <epochs> <output_dir> [batch_size] [learning_rate] [gtf_file] [vocab_file]
```

Parameters:
- `train.h5ad` (required): Training dataset with labels in `cell_type_col`
- `test.h5ad` (required): Test dataset to embed/plot
- `cell_type_col` (required): Obs column for training labels and plot colouring
- `epochs` (required): Number of training epochs
- `output_dir` (required): Parent directory; a timestamped run subdirectory is created
- `batch_size` (default: 8)
- `learning_rate` (optional; currently not used by the Python script)
- `gtf_file` (default: `gencode.v48.basic.annotation.gtf.gz`)
- `vocab_file` (default: `scgpt_vocab.txt`)

Example:

```bash
./run_finetune.sh train.h5ad test.h5ad cell_type 100 results/
```

Outputs inside run dir:
- `classes.txt`, `class_mapping.json`
- `<test_basename>_with_X_scgpt_FT.h5ad` with `.obsm['X_scgpt_FT']`
- `<test_basename>_X_scgpt_FT_umap.png`

## Data notes

- Provide `.h5ad` with a valid cell-type column; records with missing labels are dropped for training.
- Genes are mapped from Ensembl IDs to symbols via the GTF; duplicated/missing symbols are removed.
- Data are subset and ordered to the scGPT vocabulary to ensure consistent features.

## Tips and troubleshooting

- Container: ensure CUDA availability (`--nv`) and that the expected container path/binds exist.
- Memory: lower `batch_size` if you encounter OOM.
- Paths: use absolute paths for robust binding into the container.
