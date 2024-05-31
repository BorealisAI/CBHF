# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib

matplotlib.use("Agg")


def entropy_vs_feedback_percentage(env_name, expert_level):

    plt.rcParams.update({"font.size": 13})
    plt.rcParams["figure.figsize"] = (6, 5)

    entropy_thresholds = {
        "bibtex": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "media_mill": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "delicious": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }

    feedback_types = ["action_restriction_accuracy", "reward_penalty_accuracy"]
    algorithms = ["ppo", "ppo-lstm", "reinforce", "actor-critic", "linearucb"]
    colors = ["red", "green", "blue", "black", "magenta"]
    for fb_type in feedback_types:

        for idx, algo in enumerate(algorithms):

            all_hf_percentage = []
            for threshold in entropy_thresholds[env_name]:

                path_list_feedback_steps = glob.glob(
                    "results_entropies/"
                    + algo
                    + "/"
                    + env_name
                    + "/"
                    + fb_type
                    + "/*/feedback_interval_entropy/threshold_"
                    + str(threshold)
                    + "/"
                    + "Expert_Accuracy_"
                    + str(expert_level)
                    + "/*/human_feedback_steps.npy"
                )

                path_list_total_steps = glob.glob(
                    "results_entropies/"
                    + algo
                    + "/"
                    + env_name
                    + "/"
                    + fb_type
                    + "/*/feedback_interval_entropy/threshold_"
                    + str(threshold)
                    + "/"
                    + "Expert_Accuracy_"
                    + str(expert_level)
                    + "/*/total_steps.npy"
                )

                feedback_steps = np.load(path_list_feedback_steps[0])

                total_steps = np.load(path_list_total_steps[0])

                hf_percentage = (feedback_steps / total_steps) * 100

                all_hf_percentage.append(hf_percentage)

                # plotting

            plt.plot(
                entropy_thresholds[env_name],
                all_hf_percentage,
                label=algo,
                linewidth=2,
                color=colors[idx],
            )

        save_path_1 = "entropy_vs_feedback_percentage/"

        if not os.path.exists(save_path_1):
            os.makedirs(save_path_1)

        save_path_2 = save_path_1 + env_name + "/"

        if not os.path.exists(save_path_2):
            os.makedirs(save_path_2)

        save_path = save_path_2 + fb_type + "/"

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.xlabel("Entropy Thresholds")
        plt.ylabel("Percentage Expert Query")
        plt.title(env_name.upper() + " Expert Acc:" + str(expert_level))
        plt.legend()
        plt.savefig(
            save_path
            + fb_type
            + "_hf_percentage_expert_acc_"
            + str(expert_level)
            + ".pdf"
        )

        plt.clf()
        plt.close()


env_names = ["bibtex", "media_mill", "delicious"]

expert_levels = [0.3, 0.5, 0.7, 0.9]


for env_name in env_names:
    for exp_level in expert_levels:
        entropy_vs_feedback_percentage(env_name, exp_level)
