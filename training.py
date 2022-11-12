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

from game import Game


class DQN(nn.Module):
    def __init__(self, input_size=12):
        super().__init__()
        #self.fc1 = nn.Linear(2, 24)
        #self.fc2 = nn.Linear(24, 48)
        #self.fc3 = nn.Linear(48, 4)
        
        #self.fc1 = nn.Linear(input_size, 256)
        #self.fc2 = nn.Linear(256, 512)
        #self.fc3 = nn.Linear(512, 1024)
        #self.fc4 = nn.Linear(1024, 4)
        
        self.conv1 = nn.Conv2d(4, 16, (5,5), padding=2)
        self.conv2 = nn.Conv2d(16, 32, (5,5), padding=2)
        self.conv3 = nn.Conv2d(32, 48, (3,3), padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(48, 64, (3,3), padding=1)
        #self.conv4 = nn.Conv2d(24, 32, (5,5))
        
        #self.fc = nn.Linear(128, 4)
        #self.fc = nn.Linear(288, 4)
        #self.fc = nn.Linear(512, 4)
        self.fc = nn.Linear(64 * input_size, 4)
        
        
        self.leaky = nn.LeakyReLU(0.1)
        
        #self.lstm = nn.LSTM(input_size, 256, 2, dropout=0.2)
        #self.fc = nn.Linear(256, 4)
        
    def forward(self, x): 
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.2)(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.2)(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.2)(x)
        x = self.fc4(x)
        """
        
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.leaky(x)
        #x = self.pool(x)
        x = self.conv4(x)
        x = self.leaky(x)
        #x = torch.concat((torch.flatten(x, 1), crashes), dim=-1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        
        #x = self.lstm(x)
        #x = F.relu(x)
        #x = self.fc(x)
        
        
        return x
    
    
class DQNRoomSolverTraining:
            
    def __init__(self, n_episodes=30000, gamma=0.95, batch_size=128, 
                       epsilon=1.0, epsilon_min=0.1, epsilon_log_decay=0.9,
                       map_name='empty_map', max_steps=100):
        self.memory = deque(maxlen=2000)

        self.map_name = map_name
        self.max_steps = max_steps
        
        self.env = Game(visualize=False, load_map=True, map_name=self.map_name)
        m = self.env.m
        n = self.env.n
        self.input_size = ((m) // 2) * ((n) // 2)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on : ' + str(self.device))
        self.dqn = DQN(self.input_size)
        self.dqn.to(device=self.device)
        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.dqn.parameters(), lr=0.00025 )
        
    def save_model(self) :
        torch.save(self.dqn.state_dict(), 'model/' + self.map_name + '.pt')
    
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
    
    def preprocess_state(self, state):
        return [torch.tensor(np.array([state[0]]), dtype=torch.float32)]
    
    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            return random.sample(self.env.action_space, 1)[0]
        else:
            with torch.no_grad():
                return torch.argmax(self.dqn(state[0].to(self.device)).cpu()).numpy()
            
    def remember(self, state, action, reward, next_state, done):
        reward = torch.tensor(reward)
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size, e):
        y_batch, y_target_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y = self.dqn(state[0].to(self.device))
            y_target = y.clone().detach()
            with torch.no_grad():
                y_target[0][action] = reward if done else reward + self.gamma * torch.max(self.dqn(next_state[0].to(self.device))[0])
            y_batch.append(y[0])
            y_target_batch.append(y_target[0])
        
        y_batch = torch.cat(y_batch)
        y_target_batch = torch.cat(y_target_batch)
        
        self.opt.zero_grad()
        loss = self.criterion(y_batch, y_target_batch)
        loss.backward()
        self.opt.step()        
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def action_to_letter(self, action) :
        if action == 0 :
            return 'w'
        if action == 1 :
            return 'a'
        if action == 2 :
            return 's'
        if action == 3 :
            return 'd'
    
    def run(self):
        scores = deque(maxlen=100)
        j = 0
        k = 0
        k_prev = 0
        j_prev = 0
        j_max = 0
        res_list = []
        with open(os.path.join('model/results', self.map_name + '.txt'), 'w') as fp :
            for e in range(self.n_episodes):
                actions = []
                state = self.preprocess_state(self.env.reset())
                done = False
                i = 0
                while not done:
                    action = self.choose_action(state, self.get_epsilon(e))
                    next_state, reward, done = self.env.do_action(action)
                    next_state = self.preprocess_state(next_state)
                    actions.append((self.action_to_letter(action), reward))
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    i += 1
                    if i >= self.max_steps  :
                        done = True
                    if reward == 100:
                        j += 1
                    if reward == 10:
                        k += 1
                scores.append(i)
                mean_score = np.mean(scores)
                if e % 100 == 0:
                    print(actions)
                    k_prev = k - k_prev
                    j_prev = j - j_prev
                    print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
                    print('Finished ' + str(j) + ' times, up ' + str(j_prev) + ' from last time')
                    print('Killed ' + str(k) + ' enemies, up ' + str(k_prev) + ' from last time')
                    
                    if j_prev >= j_max :
                        self.save_model()
                        j_max = j_prev
                        
                    res_list.append((j_prev, k_prev))
                    fp.write(str(j_prev) + '\t' + str(k_prev) + '\n')
                        
                    k_prev = k
                    j_prev = j
                    
                self.replay(self.batch_size, e)

        return e
if __name__ == '__main__':
    agent = DQNRoomSolverTraining(map_name='bruce_lee', max_steps=150)
    agent.run()
    #agent.save_model()
    #agent.env.close()
