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
import numpy as np

# Hyperparameters


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, gamma=0, learning_rate=0.0002):
        super(ActorCritic, self).__init__()
        self.data = []
        self.gamma = gamma
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc_pi = nn.Linear(256, action_dim)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(np.array(a_lst)),
            torch.tensor(np.array(r_lst), dtype=torch.float),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(np.array(done_lst), dtype=torch.float),
        )
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + self.gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(
            self.v(s), td_target.detach()
        )

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
