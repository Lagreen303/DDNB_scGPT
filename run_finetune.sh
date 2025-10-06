#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=200G
#SBATCH --partition=quad_h200
#SBATCH --export=ALL
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24

# Check if arguments are provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <train.h5ad> <test.h5ad> <cell_type_col> <epochs> <output_dir> [batch_size] [learning_rate] [gtf_file] [vocab_file]"
    echo ""
    echo "Arguments:"
    echo "  1. train.h5ad     - Training data file"
    echo "  2. test.h5ad      - Test data file"
    echo "  3. cell_type_col  - Cell type column name"
    echo "  4. epochs         - Number of training epochs"
    echo "  5. output_dir     - Parent output directory (run subdir will be created)"
    echo "  6. batch_size     - Batch size (optional, default: 8)"
    echo "  7. learning_rate  - Learning rate (optional, default: 1e-4)"
    echo "  8. gtf_file       - GTF annotation file (optional, default: gencode.v48.basic.annotation.gtf.gz)"
    echo "  9. vocab_file     - scGPT vocabulary file (optional, default: scgpt_vocab.txt)"
    echo ""
    echo "Example:"
    echo "  $0 train.h5ad test.h5ad cell_type 100 /path/to/outdir"
    echo "  $0 train.h5ad test.h5ad cell_type 200 /path/out 16 5e-5 gencode.v48.basic.annotation.gtf.gz scgpt_vocab.txt"
    exit 1
fi

# Set default values
CELL_TYPE_COL="$3"
EPOCHS="$4"
OUTDIR="$5"
BATCH_SIZE="${6:-8}"
GTF_FILE="${7:-gencode.v48.basic.annotation.gtf.gz}"
VOCAB_FILE="${8:-scgpt_vocab.txt}"

# Run the finetuning analysis
module load apptainer
apptainer exec -H $PWD \
--bind /iridisfs/ddnb/Luke:/workspace \
--bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
--nv --fakeroot /iridisfs/ddnb/Luke/helical_sb \
python3 scgpt_finetune.py \
  --train-input "$1" \
  --test-input "$2" \
  --cell-type-col "$CELL_TYPE_COL" \
  --epochs "$EPOCHS" \
  --output "$OUTDIR" \
  --batch-size "$BATCH_SIZE" \
  --gtf "$GTF_FILE" \
  --vocab "$VOCAB_FILE"
