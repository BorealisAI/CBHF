# Copyright (c) 2024-present, Royal Bank of Canada.#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.#


from algos.ppo import PPO
import torch
from torch.distributions import Categorical
from human_feedback import HumanFeedback
import numpy as np
import copy
from algos.ppo_lstm import PPOLSTM
from algos.actor_critic import ActorCritic
from tqdm import tqdm
from algos.reinforce import Policy
import os
from scipy.stats import entropy
from algos.ee_net import EE_Net
from contextualbandits.online import LinUCB, BootstrappedTS
from algos.tamer import Tamer
from sklearn.linear_model import SGDClassifier
import warnings

warnings.simplefilter("ignore")


class RunAlgorithms(object):

    def __init__(self, env, args, logger):

        self.env = env
        self.eval_env = copy.deepcopy(env)
        self.args = args
        self.logger = logger
        self.complicated_envs = ["bibtex", "delicious", "media_mill", "eurlex", "yahoo"]
        self.device = torch.device("cpu")

        self.all_entropy_thresholds = {
            "bibtex": 5.0,
            "delicious": 6.5,
            "media_mill": 3,
            "eurlex": 6,
        }

    """
    To evaluate the policy learned by the cb algorithm
    """

    def offline_evaluation(self, model, num_runs=3, timesteps=5000):

        print("Running offline evaluation..")
        run_rewards = []

        for run in range(num_runs):

            reward_history = np.zeros(timesteps)
            ctx, ctx_idx = self.eval_env.reset()

            h_out = (
                torch.zeros([1, 1, 32], dtype=torch.float).to(self.device),
                torch.zeros([1, 1, 32], dtype=torch.float).to(self.device),
            )

            for steps in tqdm(range(timesteps)):

                if self.args.algorithm.lower() == "ee-net":
                    action = model.predict(ctx, steps)

                elif self.args.algorithm.lower() == "linearucb":
                    prob_init = model.decision_function(ctx)[0]
                    prob = prob_init / (np.sum(prob_init))
                    action = np.argmax(prob)

                elif self.args.algorithm.lower() == "ppo":
                    prob = model.pi(torch.from_numpy(ctx).float())
                    m = Categorical(prob)
                    action = m.sample().item()

                elif self.args.algorithm.lower() == "ppo-lstm":

                    h_in = h_out
                    prob, h_out = model.pi(torch.from_numpy(ctx).float(), h_in)
                    prob = prob.view(-1)
                    m = Categorical(prob)
                    action = m.sample().item()

                elif self.args.algorithm.lower() == "reinforce":
                    prob = model(torch.from_numpy(ctx).float())
                    m = Categorical(prob)
                    action = m.sample().item()

                elif self.args.algorithm.lower() == "actor-critic":
                    prob = model.pi(torch.from_numpy(ctx).float())
                    action = Categorical(prob).sample().item()

                elif self.args.algorithm.lower() == "tamer":
                    action = model.act(ctx)

                elif self.args.algorithm.lower() == "bootstrapped-ts":
                    action = model.predict(ctx)

                else:
                    raise ValueError("Invalid Algorithm During Evaluation")

                next_ctx, reward, done, ctx_idx = self.eval_env.step(action)

                reward_history[steps] = reward

                ctx = next_ctx
            run_rewards.append(reward_history)
        mean_rewards_per_run = np.mean(run_rewards, axis=0)
        std_rewards_per_run = np.std(run_rewards, axis=0)

        mean_rewards = np.zeros_like(mean_rewards_per_run)

        for i in range(1, len(mean_rewards_per_run)):
            mean_rewards[i] = np.mean(mean_rewards_per_run[: i + 1])

        return mean_rewards

    # function for training ppo algorithm
    def train_ppo(self):

        T_horizon = self.args.t_horizon

        discount = self.args.discount

        total_timesteps = self.args.timesteps

        hf_type = self.args.human_feedback.lower()

        feedback_interval = self.args.feedback_interval

        hf = HumanFeedback(self.args.env_name, self.args)

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        model = PPO(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.device,
            gamma=discount,
            learning_rate=self.args.lr_pi,
            eps_clip=self.args.noise_clip,
        ).to(self.device)

        total_steps = 0
        human_feedback_steps = 0

        for epoch in tqdm(range(total_timesteps)):

            ctx, ctx_idx = self.env.reset()

            for steps in range(T_horizon):
                suggested_action = None
                reward_penalty = None
                reward_for_query = None
                prob = model.pi(torch.from_numpy(ctx).float())

                m = Categorical(prob)

                action = m.sample().item()

                if feedback_interval != "entropy" and hf_type.lower() != "none":

                    if steps % int(feedback_interval) == 0:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )

                        human_feedback_steps += 1

                elif feedback_interval == "entropy" and hf_type.lower() != "none":

                    entropy_val = entropy(prob.data)

                    if entropy_val >= self.args.entropy_threshold:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )
                        human_feedback_steps += 1
                elif hf_type.lower() == "none":
                    pass

                else:
                    raise ValueError("Invalid feedback interval")

                if suggested_action is not None:

                    action = suggested_action

                else:
                    pass

                # taking step within the environment
                next_ctx, reward, done, ctx_idx = self.env.step(action)

                total_steps += 1

                if reward_penalty is not None:
                    reward = reward + reward_penalty
                else:
                    pass

                if reward_for_query is not None:

                    reward = reward + reward_for_query

                else:

                    pass

                model.put_data(
                    (ctx, action, reward / 100, next_ctx, prob[action].item(), done)
                )

                ctx = next_ctx

                if done:
                    ctx, idx = self.env.reset()

            model.train_net()

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

    def train_ppolstm(self):

        discount = self.args.discount
        total_timesteps = self.args.timesteps
        T_horizon = self.args.t_horizon

        hf_type = self.args.human_feedback.lower()

        feedback_interval = self.args.feedback_interval

        hf = HumanFeedback(self.args.env_name, self.args)

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        model = PPOLSTM(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.device,
            gamma=discount,
        ).to(self.device)
        total_steps = 0
        human_feedback_steps = 0
        h_out = (
            torch.zeros([1, 1, 32], dtype=torch.float).to(self.device),
            torch.zeros([1, 1, 32], dtype=torch.float).to(self.device),
        )

        for epoch in tqdm(range(total_timesteps)):

            ctx, ctx_idx = self.env.reset()

            for steps in range(T_horizon):
                suggested_action = None
                reward_penalty = None
                reward_for_query = None
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(ctx).float(), h_in)
                prob = prob.view(-1)

                m = Categorical(prob)
                action = m.sample().item()

                if feedback_interval != "entropy" and hf_type.lower() != "none":

                    if steps % int(feedback_interval) == 0:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )

                        human_feedback_steps += 1

                elif feedback_interval == "entropy" and hf_type.lower() != "none":

                    entropy_val = entropy(prob.data)

                    if entropy_val >= self.args.entropy_threshold:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )

                        human_feedback_steps += 1

                elif hf_type.lower() == "none":
                    pass

                else:
                    raise ValueError("Invalid feedback interval")

                if suggested_action is not None:

                    action = suggested_action

                else:
                    pass

                # taking step within the environment
                next_ctx, reward, done, ctx_idx = self.env.step(action)

                total_steps += 1

                if reward_penalty is not None:
                    reward = reward + reward_penalty
                else:
                    pass

                if reward_for_query is not None:

                    reward = reward + reward_for_query

                else:

                    pass

                model.put_data(
                    (
                        ctx,
                        action,
                        reward / 100,
                        next_ctx,
                        prob[action].item(),
                        h_in,
                        h_out,
                        done,
                    )
                )

                ctx = next_ctx

                if done:
                    ctx, idx = self.env.reset()

            model.train_net()

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

    def train_reinforce(self):

        T_horizon = self.args.t_horizon

        total_timesteps = self.args.timesteps

        hf_type = self.args.human_feedback.lower()

        feedback_interval = self.args.feedback_interval

        hf = HumanFeedback(self.args.env_name, self.args)

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        model = Policy(self.env.observation_space.shape[0], self.env.action_space.n)

        total_steps = 0
        human_feedback_steps = 0

        for epoch in tqdm(range(total_timesteps)):

            ctx, ctx_idx = self.env.reset()

            for steps in range(T_horizon):
                suggested_action = None
                reward_penalty = None
                reward_for_query = None

                prob = model(torch.from_numpy(ctx).float())
                m = Categorical(prob)
                action = m.sample().item()

                if feedback_interval != "entropy" and hf_type.lower() != "none":

                    if steps % int(feedback_interval) == 0:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )

                        human_feedback_steps += 1

                elif feedback_interval == "entropy" and hf_type.lower() != "none":

                    entropy_val = entropy(prob.data)

                    if entropy_val >= self.args.entropy_threshold:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )

                        human_feedback_steps += 1

                elif hf_type.lower() == "none":
                    pass

                else:
                    raise ValueError("Invalid feedback interval")

                if suggested_action is not None:

                    action = suggested_action

                else:
                    pass

                # taking step within the environment
                next_ctx, reward, done, ctx_idx = self.env.step(action)

                total_steps += 1

                if reward_penalty is not None:
                    reward = reward + reward_penalty
                else:
                    pass

                if reward_for_query is not None:

                    reward = reward + reward_for_query

                else:

                    pass

                model.put_data((reward, prob[action]))

                ctx = next_ctx

                if done:
                    ctx, _ = self.env.reset()

            model.train_net()

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

    def train_actorcritic(self):
        T_horizon = self.args.t_horizon

        total_timesteps = self.args.timesteps

        hf_type = self.args.human_feedback.lower()

        feedback_interval = self.args.feedback_interval

        hf = HumanFeedback(self.args.env_name, self.args)

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        model = ActorCritic(
            self.env.observation_space.shape[0], self.env.action_space.n
        )
        total_steps = 0
        human_feedback_steps = 0

        for epoch in tqdm(range(total_timesteps)):

            ctx, ctx_idx = self.env.reset()

            for steps in range(T_horizon):
                suggested_action = None
                reward_penalty = None
                reward_for_query = None

                prob = model.pi(torch.from_numpy(ctx).float())
                m = Categorical(prob)
                action = m.sample().item()

                if feedback_interval != "entropy" and hf_type.lower() != "none":

                    if steps % int(feedback_interval) == 0:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )

                        human_feedback_steps += 1

                elif feedback_interval == "entropy" and hf_type.lower() != "none":

                    entropy_val = entropy(prob.data)

                    if entropy_val >= self.args.entropy_threshold:

                        suggested_action, reward_penalty, reward_for_query = (
                            hf.get_feedback(
                                self.env.action_space.n, ctx, action, ctx_idx, hf_type
                            )
                        )
                        human_feedback_steps += 1
                elif hf_type.lower() == "none":
                    pass

                else:
                    raise ValueError("Invalid feedback interval")

                if suggested_action is not None:

                    action = suggested_action

                else:
                    pass

                # taking step within the environment
                next_ctx, reward, done, ctx_idx = self.env.step(action)

                total_steps += 1

                if reward_penalty is not None:
                    reward = reward + reward_penalty
                else:
                    pass

                if reward_for_query is not None:

                    reward = reward + reward_for_query

                else:

                    pass

                model.put_data((ctx, action, reward, next_ctx, done))

                ctx = next_ctx

                if done:
                    ctx, ctx_idx = self.env.reset()

            model.train_net()

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

    def train_tamer(self):

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        total_steps = 0
        human_feedback_steps = 0

        model = Tamer(self.env, 5000)

        for epi in tqdm(range(5000)):

            model._train_episode(epi)

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

    ## All CB Algorithms Below

    def train_linearucb(self):

        total_timesteps = self.args.timesteps

        hf_type = self.args.human_feedback.lower()

        feedback_interval = self.args.feedback_interval

        hf = HumanFeedback(self.args.env_name, self.args)

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        total_steps = 0
        human_feedback_steps = 0

        model = LinUCB(
            nchoices=self.env.action_space.n,
            beta_prior=None,
            alpha=0.1,
            ucb_from_empty=False,
        )
        ctx, ctx_idx = self.env.reset()

        for steps in tqdm(range(total_timesteps)):
            suggested_action = None
            reward_penalty = None
            reward_for_query = None

            prob = model.decision_function(ctx)[0] / np.sum(
                model.decision_function(ctx)[0]
            )

            action = model.predict(ctx)[0]

            if feedback_interval != "entropy" and hf_type.lower() != "none":

                if steps % int(feedback_interval) == 0:

                    suggested_action, reward_penalty, reward_for_query = (
                        hf.get_feedback(
                            self.env.action_space.n, ctx, action, ctx_idx, hf_type
                        )
                    )

                    human_feedback_steps += 1

            elif feedback_interval == "entropy" and hf_type.lower() != "none":

                entropy_val = entropy(prob)

                if entropy_val >= self.args.entropy_threshold:

                    suggested_action, reward_penalty, reward_for_query = (
                        hf.get_feedback(
                            self.env.action_space.n, ctx, action, ctx_idx, hf_type
                        )
                    )

                    human_feedback_steps += 1

            elif hf_type.lower() == "none":
                pass

            else:
                raise ValueError("Invalid feedback interval")

            if suggested_action is not None:

                action = suggested_action

            else:
                pass

            # taking step within the environment
            next_ctx, reward, done, ctx_idx = self.env.step(action)

            total_steps += 1

            if reward_penalty is not None:
                reward = reward + reward_penalty
            else:
                pass

            if reward_for_query is not None:

                reward = reward + reward_for_query

            else:

                pass

            model.partial_fit(ctx, np.array([action]), np.array([reward]))

            if done:
                ctx, ctx_idx = self.env.reset()

            ctx = next_ctx

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

        return

    def train_eenet(self):

        total_timesteps = self.args.timesteps

        hf_type = self.args.human_feedback.lower()

        feedback_interval = self.args.feedback_interval

        hf = HumanFeedback(self.args.env_name, self.args)

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        lr_1 = 0.01  # learning rate for exploitation network
        lr_2 = 0.001  # learning rate for exploration network
        lr_3 = 0.001  # learning rate for decision maker

        total_steps = 0
        human_feedback_steps = 0

        ctx, ctx_idx = self.env.reset()
        model = EE_Net(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            pool_step_size=50,
            lr_1=lr_1,
            lr_2=lr_2,
            lr_3=lr_3,
            hidden=100,
            neural_decision_maker=False,
        )

        for steps in tqdm(range(total_timesteps)):
            suggested_action = None
            reward_penalty = None
            reward_for_query = None

            # need to get the prob out of prediction
            action = model.predict(ctx, steps)

            if feedback_interval != "entropy" and hf_type.lower() != "none":

                if steps % int(feedback_interval) == 0:

                    suggested_action, reward_penalty, reward_for_query = (
                        hf.get_feedback(
                            self.env.action_space.n, ctx, action, ctx_idx, hf_type
                        )
                    )

                    human_feedback_steps += 1

            elif feedback_interval == "entropy" and hf_type.lower() != "none":

                raise ValueError("Cannot run entropy based feedback on ee-net")

            elif hf_type.lower() == "none":
                pass

            else:
                raise ValueError("Invalid feedback interval")

            if suggested_action is not None:

                action = suggested_action

            else:
                action = action

            # taking step within the environment

            next_ctx, reward, done, ctx_idx = self.env.step(action)

            model.update(ctx.reshape(1, -1), reward, steps)
            total_steps += 1

            if reward_penalty is not None:
                reward = reward + reward_penalty
            else:
                pass

            if reward_for_query is not None:

                reward = reward + reward_for_query

            else:

                pass

            if steps < 1000:
                if steps % 10 == 0:
                    loss_1, loss_2, loss_3 = model.train(steps)

            else:
                if steps % 100 == 0:
                    loss_1, loss_2, loss_3 = model.train(steps)

            ctx = next_ctx

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

        return

    def train_bootstrapped_ts(self):

        total_timesteps = self.args.timesteps

        hf_type = self.args.human_feedback.lower()

        feedback_interval = self.args.feedback_interval

        hf = HumanFeedback(self.args.env_name, self.args)

        if not os.path.exists(self.args.folder):
            os.makedirs(self.args.folder)

        total_steps = 0
        human_feedback_steps = 0
        base_algorithm = SGDClassifier()
        beta_prior_ts = ((2.0 / np.log2(self.env.action_space.n), 4), 2)
        model = BootstrappedTS(
            copy.deepcopy(base_algorithm),
            nchoices=self.env.action_space.n,
            beta_prior=beta_prior_ts,
            batch_train=True,
        )
        ctx, ctx_idx = self.env.reset()

        for steps in tqdm(range(total_timesteps)):
            suggested_action = None
            reward_penalty = None
            reward_for_query = None

            df = model.decision_function(ctx)[0]

            prob = df / df.sum()

            action = model.predict(ctx)[0]

            if feedback_interval != "entropy" and hf_type.lower() != "none":

                if steps % int(feedback_interval) == 0:

                    suggested_action, reward_penalty, reward_for_query = (
                        hf.get_feedback(
                            self.env.action_space.n, ctx, action, ctx_idx, hf_type
                        )
                    )

                    human_feedback_steps += 1

            elif feedback_interval == "entropy" and hf_type.lower() != "none":

                entropy_val = entropy(prob)

                if entropy_val >= self.args.entropy_threshold:

                    suggested_action, reward_penalty, reward_for_query = (
                        hf.get_feedback(
                            self.env.action_space.n, ctx, action, ctx_idx, hf_type
                        )
                    )

                    human_feedback_steps += 1

            elif hf_type.lower() == "none":
                pass

            else:
                raise ValueError("Invalid feedback interval")

            if suggested_action is not None:

                action = suggested_action

            else:
                pass

            # taking step within the environment
            next_ctx, reward, done, ctx_idx = self.env.step(action)

            total_steps += 1

            if reward_penalty is not None:
                reward = reward + reward_penalty
            else:
                pass

            if reward_for_query is not None:

                reward = reward + reward_for_query

            else:

                pass

            model.partial_fit(ctx, np.array([action]), np.array([reward]))

            if done:
                ctx, ctx_idx = self.env.reset()

            ctx = next_ctx

        mean_cum_reward = self.offline_evaluation(model)

        self.logger.record_data(
            mean_cum_reward,
            total_steps,
            human_feedback_steps,
        )

        self.logger.save()

        return
