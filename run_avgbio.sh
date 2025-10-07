#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=quad_h200
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

set -euo pipefail

# ---------------------------
# Usage
# ---------------------------
if [ $# -lt 3 ]; then
  echo "Usage: $0 <input.h5ad> <embedding_key> <ground_truth_col>"
  echo "Example: $0 data/test_with_X_scgpt_FT.h5ad X_scgpt_FT predicted.celltype.l2"
  exit 1
fi

INPUT=$1
EMB_KEY=$2
GT_COL=$3

BASENAME=$(basename "$INPUT" .h5ad)
IN_DIR=$(dirname "$INPUT")
OUT_DIR="${IN_DIR}/${BASENAME}"

echo "Input file       : $INPUT"
echo "Embedding key    : $EMB_KEY"
echo "Ground-truth col : $GT_COL"
echo "Output directory : $OUT_DIR (auto-created if missing)"

# ---------------------------
# Apptainer container call
# ---------------------------
module load apptainer

# Adjust this if your container path differs
CONTAINER="/iridisfs/ddnb/Luke/helical_sb"

apptainer exec -H "$PWD" \
  --bind /iridisfs/ddnb/Luke:/workspace \
  --bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
  --nv --fakeroot "$CONTAINER" \
  python3 compute_avgbio.py \
    --input "$INPUT" \
    --embedding-key "$EMB_KEY" \
    --ground-truth-col "$GT_COL"

echo "Done. Results should be at: ${OUT_DIR}/avgbio.csv"
