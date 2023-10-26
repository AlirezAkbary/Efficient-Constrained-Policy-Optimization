#!/bin/bash



eta=$1
alpha_policy=$2
alpha_lam=$3
m=$4
run=$5
gamma=$6


file_name="sMDPO.py"

python $file_name --eta $eta --alpha_policy $alpha_policy --alpha_lambd $alpha_lam --m $m --run $run --gamma $gamma