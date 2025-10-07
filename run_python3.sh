#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --partition=quad_h200
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

module load apptainer
apptainer exec -H $PWD \
--bind /iridisfs/ddnb/Luke:/workspace \
--bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
--nv --fakeroot /iridisfs/ddnb/Luke/helical_sb \
python3 $1
