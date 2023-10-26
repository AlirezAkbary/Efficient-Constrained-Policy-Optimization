#!/bin/bash


eta_list=(0.1, 1, 10)
alpha_policy_list=(0.1, 1, 10)
alpha_lambd_list=(0.1, 1)
m_list=(10, 100)
num_samples_list=(10, 20, 50)


for eta in 0.1 1 10 100 1000;
do
  for alpha_lambd in 0.0001 0.001 0.01 0.1 1;
  do
    for alpha_policy in 0.001 0.01 0.1 1;
    do
        for m in 1 10 100;
        do 
            for num_samples in 1000 2000 3000;
            do
              for run in {1..5}
              do
                sbatch run_sMDPO_TD.sh $eta $alpha_policy $alpha_lambd $m $num_samples $run
              done
            done
        done    
    done
  done
done