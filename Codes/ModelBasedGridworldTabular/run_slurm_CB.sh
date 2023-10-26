#!/bin/bash



alpha_lambd_list=(0.1, 5, 50)
num_samples_list=(10, 20, 50)



for alpha_lambd in 1 2 5 8 15 50 100 300 500;
do
    for run in {1..5}
    do
        sbatch run_CB.sh $alpha_lambd $run
    done
done


