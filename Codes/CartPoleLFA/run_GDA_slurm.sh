#!/bin/bash



alpha_lambd_list=(0.1, 5, 50)
num_samples_list=(10, 20, 50)


for alpha_pol in 0.0001 0.001 0.01 0.1;
do
    for alpha_lambd in 0.0001 0.001 0.01 0.1
    do
        for num_samples in 10 20 50;
        do
            for run in {1..5}
            do
                sbatch run_GDA.sh $alpha_lambd $num_samples $alpha_pol $run
            done
        done
    done
done

