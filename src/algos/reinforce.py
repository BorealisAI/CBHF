# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) seungeunrho and their affiliates
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
################################################################
# The code is based on the minimalRl implementation: https://github.com/seungeunrho/minimalRL
##################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, gamma=0, learning_rate=0.0002):
        super(Policy, self).__init__()
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []
