# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
import seaborn as sns


matplotlib.use("Agg")


def create_folder(f):
    return [os.makedirs(f) if not os.path.exists(f) else False]


def plot_expert_vs_perf(env_name, algorithm, feedback_interval, entropy_threshold=None):

    df = {"Feedback Types": [], "Mean Cumulative Reward": [], "Expert Accuracy": []}
    env_name = env_name.lower()

    algorithm = algorithm.lower()

    feedback_interval = feedback_interval.lower()

    feedback_types = ["action_restriction_accuracy", "reward_penalty_accuracy"]
    feedback_types_label = {
        "action_restriction_accuracy": "Action Recommendation",
        "reward_penalty_accuracy": "Reward Manipulation",
    }
    for feedback_type in feedback_types:
        feedback_type = feedback_type.lower()
        expert_accuracy = [0.1, 0.3, 0.5, 0.7, 0.9]

        for expert_acc in expert_accuracy:
            results_path = (
                "results_expert_range_v2/"
                + algorithm
                + "/"
                + env_name
                + "/"
                + feedback_type
                + "/*/feedback_interval_"
                + str(feedback_interval)
                + "/threshold_"
                + str(entropy_threshold)
                + "/Expert_Accuracy_"
                + str(expert_acc)
                + "/*/mean_cum_rewards.npy"
            )
            paths = glob.glob(results_path)

            for pth in paths:

                dat = np.load(pth)

                dat = dat.reshape(
                    -1,
                )

                dat = np.mean(dat[-20:])

                df["Feedback Types"].append(feedback_types_label[feedback_type])

                df["Mean Cumulative Reward"].append(dat)

                df["Expert Accuracy"].append(expert_acc)

    path_to_save = os.path.join("plot_expert_range", env_name, algorithm)

    df_final = pd.DataFrame(df)

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["figure.figsize"] = (9, 6)
    barplot = sns.barplot(
        x="Expert Accuracy",
        y="Mean Cumulative Reward",
        hue="Feedback Types",
        data=df_final,
    )

    for i, bar in enumerate(barplot.patches):
        bar.set_edgecolor("black")
        bar.set_linewidth(2)

    plt.legend(loc="upper left")
    plt.savefig(os.path.join(path_to_save, "expert_accuracy_range.pdf"))
    plt.clf()
    plt.close()

    print("Done")


envs = ["bibtex", "media_mill", "delicious"]
threshold = {"bibtex": 5.0, "media_mill": 3.0, "delicious": 6.5}

algorithms = [
    "ppo",
    "ppo-lstm",
    "reinforce",
    "actor-critic",
    "linearucb",
    "bootstrapped-ts",
]

for env in envs:

    for algorithm in algorithms:
        plot_expert_vs_perf(env, algorithm, "entropy", threshold[env])
