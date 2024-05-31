#!/bin/bash

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


algos=("ppo" "ppo-lstm" "reinforce" "actor-critic" "bootstrapped-ts" "linearucb")
feedback_types=("action_restriction_accuracy" "reward_penalty_accuracy")

entropy_thresholds=(1 2 3 4 5 6 7)
expert_accuracies=(0.3 0.5 0.7 0.9)
for algo in ${algos[@]}
do
    for fbtype in ${feedback_types[@]}
    do
        for exp_acc in ${expert_accuracies[@]}
        do
            for entropy in ${entropy_thresholds[@]}
            do
            python main.py --env_name bibtex --algorithm ${algo} --human_feedback ${fbtype} --feedback_interval entropy --entropy_threshold ${entropy} --expert_accuracy ${exp_acc} --folder ./results_entropies/
            done
        done
    done
done
