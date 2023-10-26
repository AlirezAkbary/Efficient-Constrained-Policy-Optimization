#!/bin/bash


file_name="GDA.py"

alpha_lam=$1

alpha_pol=$2

run=$3

gamma=$4

python $file_name --learning_rate_lambd $alpha_lam  --learning_rate_pol $alpha_pol --run $run --gamma $gamma
