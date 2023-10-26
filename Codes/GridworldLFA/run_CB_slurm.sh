#!/bin/bash


eta_list=(0.1, 1, 10)
alpha_policy_list=(0.1, 1, 10)
alpha_lambd_list=(0.1, 1)
m_list=(10, 100)
num_samples_list=(10, 20, 50)


for alpha_lambd in 0.1 0.25 1 2 5 8 15 50 100 300 500;
do
    for num_samples in 1000 2000 3000;
    do
        for tilings_size in 5;
        do
            for iht_size in 40 50 80;
            do
                for run in {1..5}
                do
                    sbatch run_CB_LSTD.sh $alpha_lambd $num_samples $tilings_size $iht_size $run
                done
            done
        done
    done  
done
