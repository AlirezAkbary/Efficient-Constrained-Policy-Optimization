#!/bin/bash

#SBATCH --mem=2048M	                      # Ask for 2 GB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00                   # The job will run for 3 hours

file_name="trainCartpole_GDA.py"
alpha_lam=$1
num_samples=$2
alpha_pol=$3
run=$4

python $file_name --learning_rate_lambd $alpha_lam --num_samples $num_samples --learning_rate_pol $alpha_pol --run $run
