#!/bin/bash
#SBATCH --mem=2048M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00                   # The job will run for 3 hours



eta=$1
alpha_policy=$2
alpha_lam=$3
m=$4
run=$5


file_name="sMDPO.py"

python $file_name --eta $eta --alpha_policy $alpha_policy --alpha_lambd $alpha_lam --m $m --run $run