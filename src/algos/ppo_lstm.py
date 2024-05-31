# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) seungeunrho and their affiliates
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
################################################################
# The code is based on the minimalRl implementation: https://github.com/seungeunrho/minimalRL
##################################################################


# PPO-LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class PPOLSTM(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        learning_rate=0.0001,
        gamma=0,
        lmbda=0.95,
        eps_clip=0.1,
        K_epoch=2,
        T_horizon=20,
    ):
        super(PPOLSTM, self).__init__()
        self.data = []

        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.T_horizon = T_horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, action_dim)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)

        x = self.fc_pi(x)

        prob = F.softmax(x, dim=2)

        return prob, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
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

        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_prime = s_prime.to(self.device)
        done_mask = done_mask.to(self.device)
        prob_a = prob_a.to(self.device)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = (
            self.make_batch()
        )

        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(self.K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage.to(self.device)
            surr2 = torch.clamp(
                ratio, 1 - self.eps_clip, 1 + self.eps_clip
            ) * advantage.to(self.device)
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
