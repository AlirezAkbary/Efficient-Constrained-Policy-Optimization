#!/bin/bash
#SBATCH --mem=2048M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00                   # The job will run for 3 hours


file_name="GDA_LSTD.py"


alpha_policy=$1
alpha_lam=$2
num_samples=$3
tilings_size=$4
iht_size=$5
run=$6

python $file_name --learning_rate_pol $alpha_policy --learning_rate_lambd $alpha_lam --num_samples $num_samples --tiling_size $tilings_size --iht_size $iht_size --run $run