# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import matplotlib

matplotlib.use("Agg")
cut_off = 1000


def create_folder(f):
    return [os.makedirs(f) if not os.path.exists(f) else False]


def plot_comparison():

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["figure.figsize"] = (10, 9)
    algorithms = ["ppo", "ppo-lstm", "reinforce", "actor-critic", "tamer"]

    entropy_threshold_label = {"bibtex": 5.0, "media_mill": 3.0, "delicious": 6.5}

    best_model_expert = {
        "bibtex": {"ppo": 0.5},
        "delicious": {"ppo": 0.3},
        "media_mill": {"reinforce": 0.9},
    }

    feedback_type = ["action_restriction_accuracy", "reward_penalty_accuracy"]

    env_names = ["bibtex", "media_mill", "delicious"]

    # generate each plot for each environment
    for env in env_names:

        baseline_path = glob.glob(
            "baseline_results/"
            + list(best_model_expert[env].keys())[0]
            + "/"
            + env
            + "/*/*/mean_cum_rewards.npy"
        )

        baseline = np.load(baseline_path[0]).reshape(
            -1,
        )

        baseline = baseline[:cut_off]

        eenet_path = glob.glob(
            "baseline_results/ee-net/" + env + "/*/*/mean_cum_rewards.npy"
        )
        eenet = np.load(eenet_path[0]).reshape(
            -1,
        )

        eenet = eenet[:cut_off]
        # model name for actio
        acc_model_name = list(best_model_expert[env].keys())[0]
        action_recommendation_path = glob.glob(
            "results_expert_range_v2/"
            + acc_model_name
            + "/"
            + env
            + "/action_restriction_accuracy/*/feedback_interval_entropy/threshold_"
            + str(entropy_threshold_label[env])
            + "/Expert_Accuracy_"
            + str(best_model_expert[env][acc_model_name])
            + "/*/mean_cum_rewards.npy"
        )

        action_recommendation_f_steps_path = glob.glob(
            "results_expert_range_v2/"
            + acc_model_name
            + "/"
            + env
            + "/action_restriction_accuracy/*/feedback_interval_entropy/threshold_"
            + str(entropy_threshold_label[env])
            + "/Expert_Accuracy_"
            + str(best_model_expert[env][acc_model_name])
            + "/*/human_feedback_steps.npy"
        )

        action_recommendation_total_steps_path = glob.glob(
            "results_expert_range_v2/"
            + acc_model_name
            + "/"
            + env
            + "/action_restriction_accuracy/*/feedback_interval_entropy/threshold_"
            + str(entropy_threshold_label[env])
            + "/Expert_Accuracy_"
            + str(best_model_expert[env][acc_model_name])
            + "/*/total_steps.npy"
        )

        action_recommendation = np.load(action_recommendation_path[0]).reshape(
            -1,
        )

        ar_percentage = (
            np.load(action_recommendation_f_steps_path[0])
            / np.load(action_recommendation_total_steps_path[0])
        ) * 100

        ar_percentage = f"{ar_percentage:.3f}"

        action_recommendation = action_recommendation[:cut_off]
        rew_model_name = list(best_model_expert[env].keys())[0]
        reward_manipulation_path = glob.glob(
            "results_expert_range_v2/"
            + rew_model_name
            + "/"
            + env
            + "/reward_penalty_accuracy/*/feedback_interval_entropy/threshold_"
            + str(entropy_threshold_label[env])
            + "/Expert_Accuracy_"
            + str(best_model_expert[env][rew_model_name])
            + "/*/mean_cum_rewards.npy"
        )

        reward_manipulation_f_steps_path = glob.glob(
            "results_expert_range_v2/"
            + rew_model_name
            + "/"
            + env
            + "/reward_penalty_accuracy/*/feedback_interval_entropy/threshold_"
            + str(entropy_threshold_label[env])
            + "/Expert_Accuracy_"
            + str(best_model_expert[env][rew_model_name])
            + "/*/human_feedback_steps.npy"
        )

        reward_manipulation_total_steps_path = glob.glob(
            "results_expert_range_v2/"
            + rew_model_name
            + "/"
            + env
            + "/reward_penalty_accuracy/*/feedback_interval_entropy/threshold_"
            + str(entropy_threshold_label[env])
            + "/Expert_Accuracy_"
            + str(best_model_expert[env][rew_model_name])
            + "/*/total_steps.npy"
        )

        reward_manipulation = np.load(reward_manipulation_path[0]).reshape(
            -1,
        )

        rm_percentage = (
            np.load(reward_manipulation_f_steps_path[0])
            / np.load(reward_manipulation_total_steps_path[0])
        ) * 100

        rm_percentage = f"{rm_percentage:.3f}"
        reward_manipulation = reward_manipulation[:cut_off]

        tamer_path = glob.glob(
            "baseline_results/tamer/" + env + "/*/*/mean_cum_rewards.npy"
        )

        tamer = np.load(tamer_path[0]).reshape(
            -1,
        )

        tamer = tamer[:cut_off]

        plt.plot(
            baseline,
            label=list(best_model_expert[env].keys())[0].upper(),
            color="blue",
            linewidth=2,
        )

        plt.plot(eenet, label="EE-Net", color="black", linewidth=2)

        if env == "bibtex":
            plt.text(650, 0.2, "HF: AR = " + str(ar_percentage) + "%")
            plt.text(650, 0.15, "HF: RM = " + str(rm_percentage) + "%")
        elif env == "media_mill":

            plt.text(650, 0.2, "HF: AR = " + str(ar_percentage) + "%")
            plt.text(650, 0.1, "HF: RM = " + str(rm_percentage) + "%")

        elif env == "delicious":
            plt.text(600, 0.2, "HF: AR = " + str(ar_percentage) + "%")
            plt.text(600, 0.1, "HF: RM = " + str(rm_percentage) + "%")

        else:
            raise ValueError("Invalid")

        plt.plot(tamer, label="TAMER", color="red", linewidth=2)

        plt.plot(
            action_recommendation,
            label=list(best_model_expert[env].keys())[0].upper() + " AR",
            color="green",
            linewidth=2,
        )

        plt.plot(
            reward_manipulation,
            label=list(best_model_expert[env].keys())[0].upper() + " RM",
            color="magenta",
            linewidth=2,
        )

        path_to_save_1 = "baseline_comparison_plots/"

        if not os.path.exists(path_to_save_1):
            os.makedirs(path_to_save_1)

        path_to_save = path_to_save_1 + env + "/"

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        plt.legend(loc="upper right")

        plt.xlabel("Eval Rounds")
        plt.ylabel("Mean Cumulative Reward")
        plt.savefig(path_to_save + env + "_comparison.pdf")

        plt.clf()
        plt.close()
        print("DONE")


plot_comparison()
