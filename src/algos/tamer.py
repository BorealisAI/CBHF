# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c)benibienz
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#########################################################
# The code is based on TAMER (https://dl.acm.org/doi/pdf/10.1145/1597735.1597738),
# implementation from (https://github.com/benibienz/TAMER).
###############################################################


import uuid


import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class SGDFunctionApproximator:
    """SGD function approximator with RBF preprocessing."""

    def __init__(self, env):

        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype="float64"
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """Returns the featurized representation for a state."""
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """

    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=0,  # only affects Q-learning
        epsilon=0,  # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        tame=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training TAMER
        output_dir=None,
        model_file_to_load=None,  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir

        # init model
        if model_file_to_load is not None:
            print(f"Loaded pretrained model: {model_file_to_load}")
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = SGDFunctionApproximator(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

    def act(self, state):
        """Epsilon-greedy Policy"""
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index):
        print(f"Episode: {episode_index + 1}  Timestep:", end="")

        tot_reward = 0
        state, state_idx = self.env.reset()

        for ts in range(200):

            # Determine next action
            action = self.act(state)

            human_reward = 1 if action == self.env.y[state_idx, action] else -1
            # Get next state and reward
            next_state, reward, done, state_idx = self.env.step(action)

            if human_reward != 0:

                self.H.update(state, action, human_reward)
                break

            tot_reward += reward
            if done:
                print(f"  Reward: {tot_reward}")
                break

            state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step
