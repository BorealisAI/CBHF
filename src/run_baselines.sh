#!/bin/bash

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


algos=("ppo" "ppo-lstm" "reinforce" "actor-critic" "nn-egreedy" "tamer" "bootstrapped-ts" "linearucb" "ee-net")

envs=("bibtex" "delicious" "media_mill")
for algo in ${algos[@]}
do
    for env in ${envs[@]}
    do
        python main.py --env_name ${env} --algorithm ${algo} --human_feedback none  --folder ./baseline_results/
        
    done
done
