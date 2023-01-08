# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:31:42 2022

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
import time
from sheepherding import Sheepherding
from training import DQN

class DQNSheepHerder:

    def __init__(self, model='model', episode=''):
        self.run_episode = True if episode != '' else False
        L = 150
        n_sheep = 10
        if self.run_episode :
            f = open(episode, 'r')
            L = int(f.readline())

            goal_split = f.readline().split(',')
            goal = [float(goal_split[0]), float(goal_split[1])]

            dog_split = f.readline().split(',')
            dog = [float(dog_split[0]), float(dog_split[1])]

            sheep = []
            n_sheep = int(f.readline().split(',')[1])
            for i in range(n_sheep) :
                sheep_split = f.readline().split(',')
                sheep.append([float(sheep_split[0]), float(sheep_split[1])])

            f.readline()
            self.actions = f.readline().split(',')[:-1]

        else :
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Running on : ' + str(self.device))

            self.model = model

        strombom_typical_values = {
            "N": n_sheep,
            "L": L,
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
            'step_strength' : 3,
            "max_steps_taken": 1000,
            "render_mode": True,
            "mode" : 'centroid'
        }
        if self.run_episode :
            strombom_typical_values['init_values'] = (goal, dog, sheep)
            self.env = Sheepherding(**strombom_typical_values)
        else :
            self.env = Sheepherding(**strombom_typical_values)
            self.dqn = DQN()
            self.load_model()
            self.dqn.to(device=self.device)

    def load_model(self):
        self.dqn.load_state_dict(torch.load('models/' + self.model, map_location=self.device))
        self.dqn.eval()

    def choose_action(self, state):
        action = torch.argmax(self.dqn(state[0].to(self.device)).cpu()).numpy().max()
        return action

    def preprocess_state(self, state):
        return [torch.tensor(np.array([state]), dtype=torch.float32)]

    def action_to_letter(self, action) :
        if action == 0 :
            return 'd'
        if action == 1 :
            return 's'
        if action == 2 :
            return 'a'
        if action == 3 :
            return 'w'

    def letter_to_action(self, action) :
        if action == 'd' :
            return 0
        if action == 's' :
            return 1
        if action == 'a' :
            return 2
        if action == 'w' :
            return 3

    def run(self):
        actions = []
        raw_state, _ = self.env.reset()
        state = self.preprocess_state(raw_state)
        done = False
        i = 0
        time.sleep(0.1)
        while not done:
            if self.run_episode :
                action = self.letter_to_action(self.actions[i])
            else :
                action = self.choose_action(state)

            i += 1
            print(str(i) + '. step | Action : ' + self.action_to_letter(action))
            next_state, reward, done, _, _ = self.env.do_action(action, True)
            next_state = self.preprocess_state(next_state)
            actions.append(self.action_to_letter(action))
            state = next_state
            print(reward)

            #time.sleep(2.5)

            if(i > 1000) :
                break



if __name__ == '__main__':
    agent = DQNSheepHerder(model="model.pt", episode='./models/episode.txt')
    agent.run()

