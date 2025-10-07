#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=quad_h200
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# --- Usage and argument parsing ---
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.h5ad> <cell_type_col> [output.png]"
    exit 1
fi

INPUT=$1
CELLTYPE_COL=$2
BASENAME=$(basename "$INPUT" .h5ad)
OUTPUT=${3:-${BASENAME}_umap.png}

echo "Input:        $INPUT"
echo "Cell type:    $CELLTYPE_COL"
echo "Output PNG:   $OUTPUT"

# --- Run inside Apptainer container ---
module load apptainer

apptainer exec -H $PWD \
  --bind /iridisfs/ddnb/Luke:/workspace \
  --bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
  --nv --fakeroot /iridisfs/ddnb/Luke/helical_sb \
  python3 plot_umaps.py "$INPUT" "$CELLTYPE_COL" "$OUTPUT"

