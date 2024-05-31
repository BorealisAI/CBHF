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


class PPO(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        learning_rate=0.0005,
        gamma=0.98,
        lmbda=0.95,
        eps_clip=0.1,
        K_epoch=3,
    ):
        super(PPO, self).__init__()
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.fc1 = nn.Linear(self.obs_dim, 256)
        self.fc_pi = nn.Linear(256, action_dim)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def pi(self, x, softmax_dim=0):
        x = x.to(self.device)
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
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = (
            torch.tensor(np.array(s_lst), dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(np.array(r_lst)),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(np.array(done_lst), dtype=torch.float),
            torch.tensor(np.array(prob_a_lst)),
        )
        self.data = []

        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_prime = s_prime.to(self.device)
        done_mask = done_mask.to(self.device)
        prob_a = prob_a.to(self.device)

        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(np.array(advantage_lst), dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage.to(self.device)
            surr2 = torch.clamp(
                ratio, 1 - self.eps_clip, 1 + self.eps_clip
            ) * advantage.to(self.device)
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                self.v(s), td_target.detach()
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
