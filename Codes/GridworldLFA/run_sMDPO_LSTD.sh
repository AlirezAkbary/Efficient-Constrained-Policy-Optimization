#!/bin/bash
#SBATCH --mem=2048M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00                   # The job will run for 3 hours


file_name="sMDPO_LSTD.py"

eta=$1
alpha_policy=$2
alpha_lam=$3
m=$4
num_samples=$5
tilings_size=$6
iht_size=$7
run=$8

python $file_name --eta $eta --alpha_policy $alpha_policy --alpha_lambd $alpha_lam --m $m --num_samples $num_samples --tiling_size $tilings_size --iht_size $iht_size --run $run