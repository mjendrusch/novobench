#!/bin/bash
#SBATCH -J novobench
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=64GB
#SBATCH -p gpu-el8
#SBATCH --gpus 1
#SBATCH -t 24:00:00
#SBATCH -C gpu=3090
#SBATCH --mail-user=michael.jendrusch@embl.de
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
#SBATCH --mail-type=FAIL,BEGIN

source $HOME/.bashrc
conda activate novobench
module load CUDA/12.0.0

python -m novobench.run $@

