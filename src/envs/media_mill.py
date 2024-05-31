# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from gym import spaces,Env
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
import os


class MediaMillEnv(Env):

    id = "media_mill-v0"

    def __init__(self):

        if not os.path.exists("envs/data/Mediamill_features.npy") or not os.path.exists(
            "envs/data/Mediamill_rewards.npy"
        ):
            self.file_directory = "envs/data/Mediamill_data.txt"
            self.X, self.y = self.parse_data(self.file_directory)

            np.save("envs/data/Mediamill_features.npy", self.X)
            np.save("envs/data/Mediamill_rewards.npy", self.y)

        else:

            self.X = np.load("envs/data/Mediamill_features.npy")
            self.y = np.load("envs/data/Mediamill_rewards.npy")

        all_low = [min(self.X[:, cols]) for cols in range(self.X.shape[1])]
        all_high = [max(self.X[:, cols]) for cols in range(self.X.shape[1])]

        self.observation_space = spaces.Box(
            low=np.array(all_low), high=np.array(all_high)
        )

        self.action_space = spaces.Discrete(self.y.shape[1])

        self.counter = np.random.randint(self.X.shape[0])
        self.start_state = self.X[self.counter, :]
        self.done = False

        self.total_correct_predictions = 0

    def step(self, action):

        self.reward = self.y[self.counter, action]
        if self.reward == 1:
            self.total_correct_predictions += 1

        self.counter = np.random.randint(self.X.shape[0])

        self.next_state = self.X[self.counter, :]

        return self.next_state, self.reward, self.done, self.counter

    def reset(self):
        self.total_correct_predictions = 0
        self.counter = np.random.randint(self.X.shape[0])

        state = self.X[self.counter, :]

        return state, self.counter

    def parse_data(self, filename):
        with open(filename, "rb") as f:
            infoline = f.readline()
            infoline = re.sub(r"^b'", "", str(infoline))
            n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
            features, labels = load_svmlight_file(
                f, n_features=n_features, multilabel=True
            )
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        features = np.array(features.todense())
        features = np.ascontiguousarray(features)
        return features, labels
