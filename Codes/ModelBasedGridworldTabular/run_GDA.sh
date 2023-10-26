#!/bin/bash
#SBATCH --mem=2048M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00                   # The job will run for 3 hours

file_name="GDA.py"

alpha_lam=$1

alpha_pol=$2

run=$3

python $file_name --learning_rate_lambd $alpha_lam  --learning_rate_pol $alpha_pol --run $run
