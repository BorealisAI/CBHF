# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import ListedColormap
import os

plt.rcParams.update({"font.size": 13})
plt.rcParams["figure.figsize"] = (6, 5)
environment_thresholds = {
    "bibtex": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "media_mill": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "delicious": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
}


env_names = ["bibtex"]

expert_accuracy_levels = [0.3, 0.5, 0.7, 0.9]

algorithms = ["linearucb"]
feedback_type = ["action_restriction_accuracy", "reward_penalty_accuracy"]
for fb_type in feedback_type:
    for env_name in env_names:

        for algo in algorithms:

            mean_val_list = []
            threshold_list = []
            expert_acc_list = []

            for t, threshold in enumerate(environment_thresholds[env_name]):

                for e, expert_acc in enumerate(expert_accuracy_levels):

                    data_path = glob.glob(
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
                        + str(expert_acc)
                        + "/*/mean_cum_rewards.npy"
                    )

                    if len(data_path) != 0:
                        data = np.load(data_path[0]).reshape(
                            -1,
                        )
                    else:
                        continue

                    data_mean = np.mean(data[-300:])

                    threshold_list.append(threshold)
                    expert_acc_list.append(expert_acc)
                    mean_val_list.append(data_mean)

            thresholds_arr = np.array(threshold_list)
            expert_acc_arr = np.array(expert_acc_list)
            mean_val_arr = np.array(mean_val_list)
            colors = mean_val_arr

            sizes_reshaped = mean_val_arr.reshape(-1, 1)

            custom_cmap = ListedColormap(["#FF5733", "#33FF57", "#3357FF"])

            scatter = plt.scatter(
                thresholds_arr,
                expert_acc_arr,
                s=mean_val_arr * 2000,
                alpha=0.6,
                c=mean_val_arr,
                cmap=custom_cmap,
                edgecolors="k",
                linewidth=2.5,
            )

            cbar = plt.colorbar(scatter)
            cbar.set_label("Color Scale")

            plt.xlabel("Entropy Thresholds")
            plt.ylabel("Expert Accuracies")

            save_path_1 = "plot_analysis_entropy_expert/"
            if not os.path.exists(save_path_1):
                os.makedirs(save_path_1)

            save_path_2 = save_path_1 + env_name + "/"

            if not os.path.exists(save_path_2):
                os.makedirs(save_path_2)

            save_path_3 = save_path_2 + fb_type + "/"

            if not os.path.exists(save_path_3):
                os.makedirs(save_path_3)

            save_path = save_path_3 + algo + "_entropy_expert.pdf"
            plt.title(algo.upper())
            plt.savefig(save_path)

            plt.clf()
            plt.close()
