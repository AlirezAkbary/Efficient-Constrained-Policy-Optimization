#!/bin/bash



alpha_lambd_list=(0.1, 5, 50)
num_samples_list=(10, 20, 50)



for alpha_lambd in 1 2 5 8 15 50 100 300 500;
do
  for num_samples in 1000 2000;
  do
    for run in {1..5}
    do
      sbatch run_CBP_TD.sh $alpha_lambd $num_samples $run
    done
  done
done


