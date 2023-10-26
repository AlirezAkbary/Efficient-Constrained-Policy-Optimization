#!/bin/bash
#SBATCH --mem=2048M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00                   # The job will run for 3 hours


file_name="CBP_LSTD.py"



alpha_lam=$1
num_samples=$2
tilings_size=$3
iht_size=$4
run=$5

python $file_name --alpha_lambd $alpha_lam --num_samples $num_samples --tiling_size $tilings_size --iht_size $iht_size --run $run