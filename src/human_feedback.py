# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from envs.bibtex import BibtexEnv
from envs.media_mill import MediaMillEnv
from envs.delicious import DeliciousEnv


COMMON_REWARD_PENALTY = -1


class HumanFeedback(object):

    def __init__(self, env_name, args):

        self.env_name = env_name
        self.args = args

        self.complicated_envs = ["bibtex", "delicious", "media_mill", "eurlex", "yahoo"]

        if env_name.lower() == "bibtex":

            self.X_vals_complicated = BibtexEnv().X
            self.y_vals_complicated = BibtexEnv().y

        elif env_name.lower() == "delicious":
            self.X_vals_complicated = DeliciousEnv().X
            self.y_vals_complicated = DeliciousEnv().y

        elif env_name.lower() == "media_mill":
            self.X_vals_complicated = MediaMillEnv().X
            self.y_vals_complicated = MediaMillEnv().y

        else:
            raise ValueError("Invalid environment name")

    def action_recommendation(
        self, num_actions, state, original_action, top_k=3, state_index=0
    ):

        acc_low = self.args.expert_accuracy

        toss = np.random.uniform(0, 1)

        if toss <= acc_low:
            # provide correct action

            final_action_pool = np.argwhere(
                self.y_vals_complicated[state_index, :]
                == np.amax(self.y_vals_complicated[state_index, :])
            )
            final_action_pool = final_action_pool.flatten()

            final_action = np.random.choice(final_action_pool)

        else:

            final_action = np.random.randint(self.y_vals_complicated.shape[1])

        return final_action

    def reward_manipulation(
        self, num_actions, state, original_action, top_k=3, state_index=0
    ):

        acc_lower = self.args.expert_accuracy
        toss = np.random.uniform(0, 1)

        if toss <= acc_lower:

            final_action_pool = np.argwhere(
                self.y_vals_complicated[state_index, :]
                == np.amax(self.y_vals_complicated[state_index, :])
            )
            final_action_pool = final_action_pool.flatten()
            reward_penalty = (
                COMMON_REWARD_PENALTY if original_action not in final_action_pool else 0
            )

        else:
            randidx = np.random.randint(self.X_vals_complicated.shape[0])

            final_action_pool = np.argwhere(
                self.y_vals_complicated[randidx, :]
                == np.amax(self.y_vals_complicated[randidx, :])
            )

            final_action_pool = final_action_pool.flatten()

            reward_penalty = (
                COMMON_REWARD_PENALTY if original_action not in final_action_pool else 0
            )

        return reward_penalty

    def get_feedback(self, num_actions, ctx, chosen_action, ctx_index, hf_type):

        penalty_received_for_querying = 0
        reward_penalty = None
        suggested_action = None

        if hf_type.lower() == "action_recommendation":

            suggested_action = self.action_recommendation(
                num_actions,
                ctx,
                chosen_action,
                ctx_index,
            )

        elif hf_type.lower() == "reward_manipulation":
            reward_penalty = self.reward_manipulation(
                num_actions,
                ctx,
                chosen_action,
                ctx_index,
            )

        else:
            raise ValueError("Invalid feedback type")

        return suggested_action, reward_penalty, penalty_received_for_querying
