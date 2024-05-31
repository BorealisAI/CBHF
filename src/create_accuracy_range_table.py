# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
import glob
import os


df = {
    "Feedback Type": [],
    "Algorithm Name": [],
    "Environment Name": [],
    "0.3": [],
    "0.5": [],
    "0.7": [],
    "0.9": [],
}


def create_folder(f):
    return [os.makedirs(f) if not os.path.exists(f) else False]


def create_table():

    env_names = ["Bibtex", "Media_Mill", "Delicious"]

    expert_accuracies = [0.3, 0.5, 0.7, 0.9]

    algorithm_names = [
        "PPO",
        "PPO-LSTM",
        "Reinforce",
        "Actor-Critic",
        "LinearUCB",
        "Bootstrapped-TS",
    ]

    entropy_levels = {"bibtex": 5.0, "media_mill": 3.0, "delicious": 6.5}

    feedback_interval = "entropy"

    feedback_types = ["action_restriction_accuracy", "reward_penalty_accuracy"]

    feedback_lables = {
        "action_restriction_accuracy": "Action Recommendation",
        "reward_penalty_accuracy": "Reward Manipulation",
    }

    for a_name in algorithm_names:

        a_name_lower = a_name.lower()

        for env_name in env_names:

            for feedback_type in feedback_types:
                df["Feedback Type"].append(feedback_lables[feedback_type])
                df["Algorithm Name"].append(a_name)
                df["Environment Name"].append(env_name)
                for expert_accuracy in expert_accuracies:

                    e_threshold = entropy_levels[env_name.lower()]
                    results_path = (
                        "results_expert_range/"
                        + a_name_lower
                        + "/"
                        + env_name
                        + "/"
                        + feedback_type
                        + "/*/feedback_interval_"
                        + str(feedback_interval)
                        + "/threshold_"
                        + str(e_threshold)
                        + "/Expert_Accuracy_"
                        + str(expert_accuracy)
                        + "/*/mean_cum_rewards.npy"
                    )

                    paths = glob.glob(results_path)

                    if len(paths) == 0:
                        dat_mean = 0
                        dat_std = 0
                    else:
                        for pth in paths:

                            dat = np.load(pth)

                            dat = dat.reshape(
                                -1,
                            )

                            dat_mean = np.mean(dat[-3000:])
                            dat_std = np.std(dat[-3000:])

                    dat_mean = f"{dat_mean:.5f}"
                    dat_std = f"{dat_std:.5f}"
                    str_dat_mean_std = str(dat_mean) + "+-" + str(dat_std)
                    df[str(expert_accuracy)].append(str_dat_mean_std)

    df_final = pd.DataFrame(df)
    df_final.to_csv("Consolidated_Algorithm_Performance.csv")
    return


create_table()
