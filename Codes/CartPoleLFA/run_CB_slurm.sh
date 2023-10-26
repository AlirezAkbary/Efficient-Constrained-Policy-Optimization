#!/bin/bash



alpha_lambd_list=(0.1, 5, 50)
num_samples_list=(10, 20, 50)



for alpha_lambd in 0.1 0.5 5 50 250 500 750 1000;
do
  for num_samples in 10 20 50;
  do
    for run in {1..5}
    do
      sbatch run_CB.sh $alpha_lambd $num_samples $run
    done
  done
done

