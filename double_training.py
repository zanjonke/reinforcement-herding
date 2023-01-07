# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:56:38 2022

@author: Kert PC
"""

# Based on: https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
from collections import deque
import random
import math
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sheepherding import Sheepherding


# from game import Game


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 4)
        #self.fc6 = nn.Linear(2048, 4)

        self.leaky = nn.LeakyReLU(0.1)

        # self.lstm = nn.LSTM(input_size, 256, 2, dropout=0.2)
        # self.fc = nn.Linear(256, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.25)(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.25)(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.25)(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.25)(x)
        x = self.fc5(x)
        #x = F.relu(x)
        #x = nn.Dropout(p=0.4)(x)
        #x = self.fc6(x)

        return x


class DQNDoubleShepherdTraining:

    def __init__(self, n_episodes=30000, gamma=0.95, batch_size=128,
                 epsilon=1.0, epsilon_min=0.1, epsilon_log_decay=0.9, max_steps=200):
        self.memory_collect = deque(maxlen=2000)
        self.memory_drive = deque(maxlen=2000)

        self.max_steps = max_steps

        # self.env = Game(visualize=False, load_map=True, map_name=self.map_name)
        strombom_typical_values = {
            "N": 40,
            "L": 150,
            "n": 10,
            "rs": 50,
            "ra": 2,
            "pa": 2,
            "c": 1.05,
            "ps": 1,
            "h": 0.5,
            "delta": 0.3,
            "p": 0.05,
            "e": 0.3,
            "delta_s": 1.5,
            "goal": [10, 10],
            "goal_radius": 30,
            "max_steps_taken": 500,
            "render_mode": False
        }
        # self.env = Game(visualize=False, load_map=True, map_name=self.map_name)
        self.env = Sheepherding(**strombom_typical_values)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.collect = True

        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on : ' + str(self.device))
        self.dqn_drive = DQN()
        self.dqn_drive.to(device=self.device)
        self.criterion = torch.nn.MSELoss()
        self.opt_drive = torch.optim.Adam(self.dqn_drive.parameters(), lr=0.00025)
        self.dqn_collect = DQN()
        self.dqn_collect.to(device=self.device)
        self.opt_collect = torch.optim.Adam(self.dqn_collect.parameters(), lr=0.00025)

    def save_models(self):
        torch.save(self.dqn_drive.state_dict(), 'models/model_d.pt')
        torch.save(self.dqn_collect.state_dict(), 'models/model_c.pt')

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return [torch.tensor(np.array([state]), dtype=torch.float32)]

    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            return random.sample(self.env.action_space, 1)[0]
        else:
            with torch.no_grad():
                if self.collect :
                    action = torch.argmax(self.dqn_collect(state[0].to(self.device)).cpu()).numpy().max()
                else :
                    action = torch.argmax(self.dqn_drive(state[0].to(self.device)).cpu()).numpy().max()
                return action

    def remember(self, state, action, reward, next_state, done):
        reward = torch.tensor(reward)
        if self.collect :
            self.memory_collect.append((state[0], action, reward, next_state[0], done))
        else :
            self.memory_drive.append((state[0], action, reward, next_state[0], done))

    def replay(self, batch_size, e):
        y_batch, y_target_batch = [], []
        if self.collect :
            minibatch = random.sample(self.memory_collect, min(len(self.memory_collect), batch_size))
        else :
            minibatch = random.sample(self.memory_drive, min(len(self.memory_drive), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y = self.dqn_collect(state[0].to(self.device)) if self.collect else self.dqn_drive(state[0].to(self.device))
            y_target = y.clone().detach()
            with torch.no_grad():
                if self.collect :
                    y_target[action] = reward if done else reward + self.gamma * torch.max(
                        self.dqn_collect(next_state[0].to(self.device)))
                else :
                    y_target[action] = reward if done else reward + self.gamma * torch.max(
                        self.dqn_drive(next_state[0].to(self.device)))
            y_batch.append(y)
            y_target_batch.append(y_target)

        y_batch = torch.cat(y_batch)
        y_target_batch = torch.cat(y_target_batch)

        if self.collect :
            self.opt_collect.zero_grad()
            loss = self.criterion(y_batch, y_target_batch)
            loss.backward()
            self.opt_collect.step()
        else :
            self.opt_drive.zero_grad()
            loss = self.criterion(y_batch, y_target_batch)
            loss.backward()
            self.opt_drive.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def action_to_letter(self, action):
        if action == 0:
            return 'd'
        if action == 1:
            return 's'
        if action == 2:
            return 'a'
        if action == 3:
            return 'w'

    def run(self):
        scores = deque(maxlen=100)
        steps = deque(maxlen=100)
        max_asng = 0
        with open(os.path.join('models/results.txt'), 'w') as fp:
            for e in range(self.n_episodes):
                actions = []
                state = self.preprocess_state(self.env.reset())
                done = False
                self.collect = True
                i = 0
                max_sheep = 0
                while not done:
                    action = self.choose_action(state, self.get_epsilon(e))
                    next_state, reward, done, sheep_near_goal, deformation = self.env.do_action(action, self.collect)
                    next_state = self.preprocess_state(next_state)
                    actions.append((self.action_to_letter(action), reward, self.collect))
                    self.remember(state, action, reward, next_state, done)
                    state = next_state

                    if max_sheep < sheep_near_goal :
                        max_sheep = sheep_near_goal

                    #self.collect = False if deformation > -2 else True

                    i += 1
                    if i >= self.max_steps:
                        done = True
                        scores.append(max_sheep)

                    if done:
                        steps.append(i)
                if e % 100 == 0:
                    print(actions)
                    print(actions[-10:])
                    asng = sum(scores) / 100
                    print('ASNG metric at ' + str(e + 1) + '. episode: ' + str(asng) + ' | Avg. steps : ' + str(
                        sum(steps) / 100))
                    if asng >= max_asng:
                        self.save_models()
                        max_asng = asng
                    fp.write(str(asng) + '\n')

                self.replay(self.batch_size, e)

        return e


if __name__ == '__main__':
    agent = DQNDoubleShepherdTraining()
    agent.run()
    # agent.save_model()
    # agent.env.close()
