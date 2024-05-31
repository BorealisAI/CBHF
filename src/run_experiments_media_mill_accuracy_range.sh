#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


algos=("ppo" "ppo-lstm" "reinforce" "actor-critic" "linearucb" "bootstrapped-ts")
feedback_types=("action_restriction_accuracy" "reward_penalty_accuracy")

expert_accuracies=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
for algo in ${algos[@]}
do
    for fbtype in ${feedback_types[@]}
    do
        for exp_acc in ${expert_accuracies[@]}
        do
        python main.py --env_name media_mill --algorithm ${algo} --human_feedback ${fbtype} --feedback_interval entropy --entropy_threshold 3.0 --expert_accuracy ${exp_acc} --folder ./results_expert_range_v2/
        done
    done
done
