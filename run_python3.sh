#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=470G
#SBATCH --partition=quad_h200
#SBATCH --export=ALL
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24

module load apptainer
apptainer exec -H $PWD \
--bind /iridisfs/ddnb/Luke:/workspace \
--bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
--nv --fakeroot /iridisfs/ddnb/Luke/helical_sb \
python3 $1
