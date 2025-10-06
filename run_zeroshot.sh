#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=200G
#SBATCH --partition=quad_h200
#SBATCH --export=ALL
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.h5ad> <output_dir> [gtf_file] [vocab_file] [cell_type_column]"
    echo ""
    echo "Arguments:"
    echo "  1. input.h5ad      - Input data file"
    echo "  2. output_dir      - Output directory"
    echo "  3. gtf_file        - GTF annotation file (optional, default: gencode.v48.basic.annotation.gtf.gz)"
    echo "  4. vocab_file      - scGPT vocabulary file (optional, default: scgpt_vocab.txt)"
    echo "  5. cell_type_col   - Cell type column name (optional, default: cell_type)"
    echo ""
    echo "Example:"
    echo "  $0 data.h5ad results/"
    echo "  $0 data.h5ad results/ gencode.v48.basic.annotation.gtf.gz scgpt_vocab.txt cell_type"
    exit 1
fi

# Set default values
GTF_FILE="${4:-gencode.v48.basic.annotation.gtf.gz}"
VOCAB_FILE="${5:-scgpt_vocab.txt}"
CELL_TYPE_COL="${3:-cell_type}"

# Run the zeroshot analysis
module load apptainer
apptainer exec -H $PWD \
--bind /iridisfs/ddnb/Luke:/workspace \
--bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
--nv --fakeroot /iridisfs/ddnb/Luke/helical_sb \
python3 scgpt_zeroshot.py \
  --input "$1" \
  --output "$2" \
  --cell-type-col "$CELL_TYPE_COL"\
  --gtf "$GTF_FILE" \
  --vocab "$VOCAB_FILE" \
  
