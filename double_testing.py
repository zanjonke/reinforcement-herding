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
from double_training import DQN

class HierarchicalDQNSheepHerder:

    def __init__(self, model_d='model_d',model_c="model_c", episode=''):
        self.run_episode = True if episode != '' else False
        self.collect = True
        self.deformation_limit = -2
        L = 100
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

            self.model_d = model_d
            self.model_c = model_c

        strombom_typical_values = {
            "N": 10,
            "L": 80,
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
            "max_steps_taken": 998,
            "step_strength": 3,
            "render_mode": True,
            "mode" : 'centroid'
        }
        if self.run_episode :
            strombom_typical_values['init_values'] = (goal, dog, sheep)
            strombom_typical_values['max_steps_taken'] = len(self.actions) - 1
            self.env = Sheepherding(**strombom_typical_values)
        else :
            self.env = Sheepherding(**strombom_typical_values)
            self.dqn_c = DQN()
            self.dqn_d = DQN()
            self.load_models()
            self.dqn_c.to(device=self.device)
            self.dqn_d.to(device=self.device)

    def load_models(self):
        self.dqn_c.load_state_dict(torch.load(self.model_c, map_location=self.device))
        self.dqn_c.eval()

        self.dqn_d.load_state_dict(torch.load(self.model_d, map_location=self.device))
        self.dqn_d.eval()

    def choose_action(self, state):        
        if self.collect:
            action = torch.argmax(self.dqn_c(state[0].to(self.device)).cpu()).numpy().max()
        else:
            action = torch.argmax(self.dqn_d(state[0].to(self.device)).cpu()).numpy().max()
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
            
            next_state, reward, done, _, deformation = self.env.do_action(action, True)
            print(str(i) + '. step | Action : ' + self.action_to_letter(action) + " | Done: " + str(done))
            #print("deformation: " + str(deformation))
            next_state = self.preprocess_state(next_state)
            actions.append(self.action_to_letter(action))
            state = next_state
            #print(reward)
            self.collect = False if deformation > self.deformation_limit else True

            #time.sleep(2.5)

            #if(i > 500) :
            #    break
            if done:
                break

        #self.env.store_frames_in_mp4()
        self.env.store_frames_in_gif("example_double_testing.gif")


if __name__ == '__main__':
    model_c_path = "models/backup/model_c_centroid.pt"
    model_d_path = "models/backup/model_d_centroid.pt"
    
    #agent = HierarchicalDQNSheepHerder(model_d=model_d_path,model_c=model_c_path, episode='./models/episode.txt')
    agent = HierarchicalDQNSheepHerder(model_d=model_d_path,model_c=model_c_path, episode='')
    agent.run()

