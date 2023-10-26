#!/bin/bash
#SBATCH --mem=2048M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00                   # The job will run for 3 hours

file_name="CBP.py"

alpha_lam=$1
run=$2

python $file_name --alpha_lambd $alpha_lam --run $run
