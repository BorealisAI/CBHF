# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import argparse
import os
import time
import json


def get_parser():
    parser = argparse.ArgumentParser()

    named_args = parser.add_argument_group("named arguments")

    named_args.add_argument(
        "--seed",
        "--seed",
        help="""# Sets Gym, PyTorch and Numpy seeds""",
        required=False,
        type=int,
        default=0,
    )

    named_args.add_argument(
        "--algorithm",
        "--algorithm",
        help="""# Sets Gym, PyTorch and Numpy seeds""",
        required=False,
        type=str,
        default="ppo",
    )

    named_args.add_argument(
        "--env_name",
        "--env_name",
        help="""recommendation environment name""",
        required=False,
        type=str,
        default="heart_disease",
    )

    named_args.add_argument(
        "--timesteps",
        "--timesteps",
        help="""recommendation environment name""",
        required=False,
        type=int,
        default=5000,
    )

    named_args.add_argument(
        "--eval_freq",
        "--eval_freq",
        help="""Frequency where separate montecarlo evaluations are run""",
        required=False,
        type=float,
        default=10,
    )

    named_args.add_argument(
        "--t_horizon",
        "--t_horizon",
        help="""# Max time steps to run environment for""",
        required=False,
        type=int,
        default=20,
    )

    named_args.add_argument(
        "--save_models",
        "--save_models",
        help="""# Whether or not models are saved""",
        type=bool,
        default=True,
    )

    named_args.add_argument(
        "--discount",
        "--discount",
        help="""# Discount factor""",
        required=False,
        type=float,
        default=0,
    )

    named_args.add_argument(
        "--noise_clip",
        "--noise_clip",
        help="""# Range to clip target policy noise""",
        required=False,
        type=float,
        default=0.1,
    )

    named_args.add_argument(
        "--lambda_critic",
        "--lambda_critic",
        help="""# Lambda trade-off for critic regularizer""",
        required=False,
        type=float,
        default=0.95,
    )

    named_args.add_argument(
        "--batch_size",
        "--batch_size",
        help="""Batch size for the sac""",
        required=False,
        type=int,
        default=32,
    )

    named_args.add_argument(
        "--lr_pi",
        "--lr_pi",
        help="Learning rate of the policy network",
        required=False,
        type=float,
        default=0.0005,
    )

    named_args.add_argument(
        "--lr_q",
        "--lr_q",
        help="Learning rate of the q network in sac",
        required=False,
        type=float,
        default=0.001,
    )

    named_args.add_argument(
        "-f",
        "--folder",
        help="""Folder to save data to""",
        required=False,
        type=str,
        default="./results/",
    )

    named_args.add_argument(
        "--human_feedback",
        help="type of human feedback to provide",
        required=True,
        type=str,
        default="None",
    )

    named_args.add_argument(
        "--feedback_interval",
        help="interval at which feedback should be obtained",
        required=False,
        type=str,
        default=1,
    )

    named_args.add_argument(
        "--entropy_threshold",
        help="setting the threshold of the entropy based feedback",
        required=False,
        type=float,
        default=5,
    )

    named_args.add_argument(
        "--expert_accuracy",
        help="setting the accuracy for the expert",
        required=False,
        type=float,
        default=None,
    )

    return parser


def create_folder(f):
    return [os.makedirs(f) if not os.path.exists(f) else False]


class Logger(object):
    def __init__(
        self,
        args,
        experiment_name="",
        environment_name="",
        seed="",
        human_feedback="",
        feedback_interval="",
        folder="./results",
    ):
        """
        Saves experimental metrics for use later.
        :param experiment_name: name of the experiment
        :param folder: location to save data
        : param environment_name: name of the environment
        """
        self.mean_reward = []
        self.total_reward = []
        self.mean_accuracy = []

        if human_feedback.lower() == "none":
            self.save_folder = os.path.join(
                folder,
                experiment_name,
                environment_name,
                seed,
                time.strftime("%y-%m-%d-%H-%M-%s"),
            )

        elif (
            human_feedback.lower() == "action_recommendation"
            or human_feedback.lower() == "reward_manipulation"
        ):
            accuracy = args.expert_accuracy

            if feedback_interval.lower() == "entropy":
                self.save_folder = os.path.join(
                    folder,
                    experiment_name,
                    environment_name,
                    human_feedback.lower(),
                    seed,
                    "feedback_interval_" + str(feedback_interval),
                    "threshold_" + str(args.entropy_threshold),
                    "Expert_Accuracy_" + str(accuracy),
                    time.strftime("%y-%m-%d-%H-%M-%s"),
                )
            else:
                self.save_folder = os.path.join(
                    folder,
                    experiment_name,
                    environment_name,
                    human_feedback.lower(),
                    seed,
                    "feedback_interval_" + str(feedback_interval),
                    time.strftime("%y-%m-%d-%H-%M-%s"),
                )
        create_folder(self.save_folder)

    def record_reward(self, reward_return):
        self.returns_eval = reward_return

    def record_data(
        self,
        mean_reward,
        total_steps,
        human_feedback_steps,
    ):

        self.mean_reward.append(mean_reward)

        self.total_steps = total_steps
        self.human_feedback_steps = human_feedback_steps

    def save(self):
        np.save(
            os.path.join(self.save_folder, "mean_cum_rewards.npy"), self.mean_reward
        )

        np.save(os.path.join(self.save_folder, "total_steps.npy"), self.total_steps)

        np.save(
            os.path.join(self.save_folder, "human_feedback_steps.npy"),
            self.human_feedback_steps,
        )

    def save_args(self, args):
        """
        Save the command line arguments
        """
        with open(os.path.join(self.save_folder, "params.json"), "w") as f:
            json.dump(dict(args._get_kwargs()), f)
