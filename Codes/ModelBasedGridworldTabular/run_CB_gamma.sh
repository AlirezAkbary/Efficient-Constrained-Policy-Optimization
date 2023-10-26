#!/bin/bash


file_name="CBP.py"

alpha_lam=$1
run=$2
gamma=$3

python $file_name --alpha_lambd $alpha_lam --run $run --gamma $gamma
