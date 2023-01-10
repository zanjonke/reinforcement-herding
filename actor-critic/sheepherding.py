import cv2 as cv
import numpy as np
from uuid import uuid4
from scipy.spatial.distance import cdist
from pprint import pprint
from numpy import linalg as LA
import random
#import matplotlib.pyplot as plt
import math
#np.random.seed(1)
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import torch
import gym

from typing import Optional, Union
import os
from gym.envs.registration import make, register, registry, spec
from shutil import rmtree
from collections import deque
import imageio

class Dog:
    
    def __init__(self, **params):
        self.position = params["starting_position"]
        self.id = str(uuid4())
        self.delta_s = 0.1
        self.target_position = np.array([150,150])
        self.e= params["e"]
        self.behind_flock = params["ra"]*np.sqrt(params["N"]) 
        self.behind_sheep = params["ra"]*10 #???
        self.ra = params["ra"]
        self.N = params["N"]
        #self.step_strength = params["step_strength"]
        self.step_strength = 7
        
        self.possible_actions = [0, 1, 2, 3, 4,5,6,7]
        self.step_vectors = {
            0: [0,1],
            1: [1,0],
            2: [0,-1],
            3: [-1,0],
            4: [1,1],
            5: [1,-1],
            6: [-1,1],
            7: [-1,-1]
        }

    def set_position(self, position):
        self.position = position

    # move the sheep
    # does not need to return a reward
    def step(self, action, L):
        if not action in self.possible_actions:
            return

        move_vector = np.array(self.step_vectors[action])
        new_position = self.position + move_vector*self.step_strength
        
        if np.logical_or(new_position<2, new_position>(L-3)).any():
            return 
        self.position = new_position


    """
    def calc_GCM(self, sheep_positions):
        return np.mean([sheep_positions[sheep_id] for sheep_id in sheep_positions],axis=0)

    def calc_sheep_dists_to_GCM(self, GCM, sheep_positions):
        sheep_pos = np.array([sheep_positions[sheep_id] for sheep_id in sheep_positions])
        sheep_dists_to_GCM = cdist(sheep_pos, [GCM])
        sheep_ids = list(sheep_positions.keys())
        return {sheep_ids[idxi]: row[0] for idxi, row in enumerate(sheep_dists_to_GCM)}

    def strombom_step(self, sheep_positions, sheep_dists):
        GCM = self.calc_GCM(sheep_positions)
        sheep_dists_to_GCM = self.calc_sheep_dists_to_GCM(GCM, sheep_positions)
        farthest_sheep = max(sheep_dists_to_GCM, key=sheep_dists_to_GCM.get)
        print("farthest_sheep: " + str(farthest_sheep))
        fN = self.ra*(self.N**(2/3))
        if sheep_dists_to_GCM[farthest_sheep] < fN:
            Pd = GCM - self.target_position
            Pd = ???
        else:
            Pc = 1.5*(GCM-sheep_dists_to_GCM[farthest_sheep])
            
    """
    def get_position(self):
        return self.position

class Sheep:
    
    

    def __init__(self, **params):
        self.id=params["_id"]
        self.position = params["starting_position"]
        self.n = params["n"]
        self.rs = params["rs"]
        self.ra = params["ra"]
        self.pa = params["pa"]
        self.c = params["c"]
        self.ps = params["ps"]
        self.h = params["h"]
        self.delta = params["delta"]
        self.p = params["p"]
        self.e = params["e"]
        self.inertia=np.array([0,0])

    def set_position(self, position):
        self.position = position
    
    def calc_LCM(self, sheep_positions, sheep_dists):        
        closest_sheep = sorted(sheep_dists, key=sheep_dists.get)[1:self.n+1]        
        closest_sheep_positions = [sheep_positions[sheep_id] for sheep_id in closest_sheep]        
        return np.mean(closest_sheep_positions, axis=0)

    def calc_repulsion_force_vector_from_too_close_neighborhood(self,sheep_positions,sheep_dists):
        too_close_sheep = list(filter(lambda sheep_id: sheep_dists[sheep_id]<self.ra and sheep_id != self.id, sheep_dists))
        if len(too_close_sheep)==0:
            return np.array([0,0])    
        
        Ra = [(self.position-sheep_positions[sheep_id])/(LA.norm(self.position-sheep_positions[sheep_id])) for sheep_id in too_close_sheep]
        
        return np.mean(Ra,axis=0)
        

    def calc_attraction_force_to_closest_n_sheep(self, sheep_positions, sheep_dists):
        LCM = self.calc_LCM(sheep_positions, sheep_dists)
        return LCM - self.position

    def calc_repulsion_force_vector_from_dog(self,dog_position, dog_dist):        
        if dog_dist > self.rs:
            return np.array([0,0])

        return self.position - dog_position

    def graze(self):
        np.random.seed(0)
        self.position += np.random.uniform(low=-1, high=1, size=2)*0.1


    # move the sheep
    # does not need to return a reward
    def step(self,sheep_positions, sheep_dists, dog_position, dog_dist, L):                                                        
        
        if dog_dist < self.rs:        
            C = self.calc_attraction_force_to_closest_n_sheep(sheep_positions, sheep_dists)
            C = C/LA.norm(C)

            Ra = self.calc_repulsion_force_vector_from_too_close_neighborhood(sheep_positions, sheep_dists)
            if not np.array_equal(Ra, np.array([0,0])):
                Ra = Ra/LA.norm(Ra)

            Rs = self.calc_repulsion_force_vector_from_dog(dog_position, dog_dist)        
            if not np.array_equal(Rs, np.array([0,0])):
                Rs = Rs/LA.norm(Rs)

            E = np.random.uniform(low=-1, high=1, size=2)*0.1
            
            H = self.h*self.inertia + self.c*C + self.pa*Ra + self.ps*Rs + self.e*E
            
            new_position = self.position + self.delta*H
            if np.logical_or(new_position<2, new_position>(L-3)).any():
                return 

            self.position = new_position
            self.inertia = H 
            
        else:
            self.graze()
            self.inertia=np.array([0,0])

    def get_position(self):
        return self.position
    

class Sheepherding(gym.Env[np.ndarray, int]):
    

    def __init__(self, render_mode: Optional[str] = None):
        params = {
            "N":30,
            "L":100,
            "n":30,
            "rs":50,
            "ra":2,
            "pa":2,
            "c":1.05,
            "ps":1,
            "h":0.5,
            "delta":1.5,
            "p":0.05,
            "e":0.3,
            "delta_s":1.5,
            "goal":[30,30],
            "goal_radius":30,
            "max_steps_taken":300,
            "render_mode":False,
        }
        self.N = params["N"]                 # number of sheep
        self.L = params["L"]                 # size of the grid

        self.n = params["n"]                 # number of nearest neighbors to consider                         -> relevant for the sheep
        self.rs = params["rs"]               # sheperd detection distance                                      -> relevant for the sheep
        self.ra = params["ra"]               # agent to agent interaction distance                             -> relevant for the sheep
        self.pa = params["pa"]               # relative strength of repulsion from other agents                -> relevant for the sheep
        self.c = params["c"]                 # relative strength of attraction to the n nearest neighbors      -> relevant for the sheep
        self.ps = params["ps"]               # relative strength of repulsion from the sheperd                 -> relevant for the sheep
        self.h = params["h"]                 # relative strength of proceeding in the previous direction       -> relevant for the sheep        
        self.delta = params["delta"]         # agent displacement per time step                                -> relevant for the sheep
        self.p = params["p"]                 # probability of moving per time step while grazing               -> relevant for the sheep

        self.e = params["e"]                 # relative strength of angular noise                              -> relevant for the sheep and the sheperd

        self.delta_s = params["delta_s"]     # sheperd displacement per time step                              -> relevant for the sheperd
        
        self.sheep = None
        self.dog = None
        self.goal = params["goal"]
        self.goal_radius = params["goal_radius"]
        self.random_init()
        self.steps_taken=0
        self.max_steps_taken = params["max_steps_taken"]
        #self.action_space = [0,1,2,3]
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 400, 400), dtype=np.uint8)
        self.render_mode = params["render_mode"]
        self.store_video_file_name = params.get("store_video_file_name", "sheepherding_game.mp4")        
        self.step_strength = params.get("step_strength", 1)
        self.frames = []
        self.current_reward = 0
        self.done = False
        self.number_of_states_of_unherded_sheep = 0
        self.number_of_states_not_closer = 0

        self.eps = 0.001
        self.last_n_dists = deque(maxlen=5)
    


        
    def close(self):
        return

    def __str__(self):
        return "Sheepherding"

    def store_frames_in_gif(self,filename=None):                      
        imageio.mimsave(filename, self.frames)

    def store_frames_in_mp4(self,filename=None):              
        
        if len(self.frames)==0:
            return
        
        if filename is None:
            filename = self.store_video_file_name
        
        # Define the codec and create VideoWriter object
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        

        frameshape = self.frames[0].shape[0:2]
        #print("frameshape: " + str(frameshape))
        #print("filename: " + str(filename))
        out = cv.VideoWriter(filename,fourcc, 20, (210,160))
        folder_path = "/".join(filename.split("/")[:-1])
        for idx, frame in enumerate(self.frames):            
            
            #frame = np.moveaxis(frame, 0, 2)
            
            #cv.imshow("frame", frame)
            #cv.waitKey(0)
            
            #cv.imwrite(folder_path + "/"+str(idx)+".png", frame)
            
            #frame_read = cv.imread("games/"+str(idx)+".png")
            #print("")
            #plt.imshow(frame)
            #plt.show()
            #cv.imshow("frame", frame)
            #cv.waitKey(0)
            print("frameshape: " + str(frameshape))
            print("frame.shape: " + str(frame.shape))
            print("frame.dtype: " + str(frame.dtype))
            print("out: " + str(out))
            print("type(out): " + str(type(out)))
            pprint(dir(out))
            pprint(out.__sizeof__())
            exit()
            #print("frame_read.shape: " + str(frame_read.shape))
            #print("frame_read.dtype: " + str(frame_read.dtype))
            #print(frame)
            #out.write(frame)
            #print("r: " + str(r))
        # Release everything if job is finished    
        # 
        
        #print("Stored gamefile: " + str(folder_path) + ", " + str(filename))    
        #print("os.path.exists(folder_path): " + str(os.path.exists(folder_path)))    
        #print("os.path.exists(filename): " + str(os.path.exists(filename)))            
        
        out.release()     


    def reset(self, **kwargs):
        self.current_reward = 0
        self.number_of_states_of_unherded_sheep = 0
        self.steps_taken=0
        self.random_init()
        self.frames = []
        GCM = self.calc_GCM()
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
        deformation = sum([-1 if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_centroid]) 
        #obs = self.calc_centroid_based_observation_vector()
        self.render()        
        
        return np.array(self.frames[-1], dtype=np.uint8)
        #return obs


    # randomly initialize state
    def random_init(self):
        #np.random.seed(0)
        random_x = np.random.uniform(low=0.4,high=0.95, size=self.N)*self.L
        random_y = np.random.uniform(low=0.4,high=0.95, size=self.N)*self.L
        self.sheep = [
            Sheep(
                _id=i, 
                starting_position=np.array([random_x[i], random_y[i]]), 
                ra=self.ra, 
                rs=self.rs, 
                n=self.n,
                pa=self.pa,
                c=self.c,
                ps=self.ps,
                h=self.h,
                delta=self.delta,
                p=self.p,
                e=self.e
            ) for i in range(0,self.N)]
        if np.random.random() < 1:
            dog_starting_position = [90, 90]
        else:
            dog_starting_position = [40, 40]
        self.dog = Dog(starting_position=dog_starting_position, ra=self.ra, N=self.N, e=self.e,step_strength=2)
        #self.dog = Dog(starting_position=list(np.random.uniform(low=0,high=0.5, size=2)*self.L), ra=self.ra, N=self.N, e=self.e)
        #self.goal = list(np.random.uniform(low=0,high=0.5, size=2)*self.L)

    def calc_GCM(self):        
        return np.mean([sheep.get_position() for sheep in self.sheep],axis=0)

    def distance(self, p1, p2):
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    #def get_sheep_out_of_group(self):
    #    sheep_positions = [sheep.get_position() for sheep in self.sheep]    
    #    
    #    GCM = self.calc_GCM()
    #    max_dist_from_GCM = self.ra*(self.N**(2/3))
    #    sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
    #    return [idx for idx, dist in enumerate(sheep_dists_from_centroid) if dist > max_dist_from_GCM] # negative reward
    
    def calc_reward(self):

        max_dist_in_map = np.sqrt(self.L**2 + self.L**2)
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        total_reward = 0
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
        sheep_dists_from_centroid = [dist[0] for dist in sheep_dists_from_centroid]
        total_reward += sum([-1 if dist > max_dist_from_GCM else 1 for dist in sheep_dists_from_centroid]) # negative reward

        deformation = total_reward                        
        
        if deformation == 0:    
                 
            GCM_to_goal_distance = cdist([GCM], [self.goal])[0][0]
            if GCM_to_goal_distance == 0:
                GCM_to_goal_distance = 0.1

            GCM_to_goal_distance = (GCM_to_goal_distance/max_dist_in_map)**(-1)

            dog_to_goal_distance = cdist([self.goal], [self.dog.get_position()])[0][0]
            if dog_to_goal_distance == 0:
                dog_to_goal_distance = 0.1
            dog_to_goal_distance = (dog_to_goal_distance/max_dist_in_map)**(-1)

            dog_to_GCM_distance = cdist([GCM], [self.dog.get_position()])[0][0]
            if dog_to_GCM_distance == 0:
                dog_to_GCM_distance = 0.1
            dog_to_GCM_distance = (dog_to_GCM_distance/max_dist_in_map)**(-1)

            #print("GCM_to_goal_distance: " + str(GCM_to_goal_distance))
            #total_reward += (GCM_to_goal_distance + dog_to_goal_distance + dog_to_GCM_distance)
            #total_reward += GCM_to_goal_distance
                            
        return total_reward                                
    
    def calc_centroid_based_observation_vector(self):        
        sheep_pos = [sheep.get_position() for sheep in self.sheep]        
        sheep_pos_dict = {sheep.id: sheep.get_position() for sheep in self.sheep}
        sheep_centroid  = np.mean(sheep_pos,axis=0)        
        
        sheep_dists_from_centroid = cdist(sheep_pos, [sheep_centroid])
        sheep_dists_from_centroid_dict = {self.sheep[idxi].id: row[0] for idxi, row in enumerate(sheep_dists_from_centroid)}        
        farthest_sheep_from_centroid_id = max(sheep_dists_from_centroid_dict, key=sheep_dists_from_centroid_dict.get)                
        closest_sheep_from_centroid_id = min(sheep_dists_from_centroid_dict, key=sheep_dists_from_centroid_dict.get)                
        farthest_sheep_from_centroid_position = sheep_pos_dict[farthest_sheep_from_centroid_id]                
        closest_sheep_from_centroid_position = sheep_pos_dict[closest_sheep_from_centroid_id]                        

        sheep_dists_from_goal = cdist(sheep_pos, [self.goal])
        sheep_dists_from_goal_dict = {self.sheep[idxi].id: row[0] for idxi, row in enumerate(sheep_dists_from_goal)} 
        farthest_sheep_from_goal_id = max(sheep_dists_from_goal_dict, key=sheep_dists_from_goal_dict.get)                
        closest_sheep_from_goal_id = min(sheep_dists_from_goal_dict, key=sheep_dists_from_goal_dict.get)                
        farthest_sheep_from_goal_position = sheep_pos_dict[farthest_sheep_from_goal_id]                
        closest_sheep_from_goal_position = sheep_pos_dict[closest_sheep_from_goal_id]                               

        dog_position = self.dog.get_position()
        goal_position = self.goal
        obs_vector = np.concatenate([sheep_centroid, farthest_sheep_from_centroid_position, closest_sheep_from_centroid_position,farthest_sheep_from_goal_position, closest_sheep_from_goal_position, dog_position, goal_position],axis=0)
        return obs_vector


    def calc_number_of_sheep_not_hered(self):        
        sheep_positions = [sheep.get_position() for sheep in self.sheep]            
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_GCM = cdist(sheep_positions, [GCM])
        sheep_dists_from_GCM = [dist[0] for dist in sheep_dists_from_GCM]
        return sum([1 if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_GCM]) 


    def calc_GCM_dist_to_goal(self):     
        #print("calc_GCM_dist_to_goal")   
        GCM = self.calc_GCM()
        GCM_to_goal_distance = cdist([GCM], [self.goal], "cityblock")[0][0]                        
        #print("calc_GCM_dist_to_goal: " + str(GCM_to_goal_distance))   
        return GCM_to_goal_distance

    def calc_number_of_sheep_near_goal(self):
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        sheep_dists_from_goal = cdist(sheep_positions, [self.goal])
        num_of_sheep_close_to_goal = sum([1 if dist < self.goal_radius else 0 for dist in sheep_dists_from_goal])
        return num_of_sheep_close_to_goal 

    # move the dog in the wanted position
    # and change the position of the sheep accordingly
    # also return the next state, and reward for this action    
    #def do_action(self, action, collect=True):        
    def step(self, action):     
        #print("")
        self.steps_taken += 1

        not_herded_sheep_before = self.calc_number_of_sheep_not_hered()
        herded_sheep_before = self.N - not_herded_sheep_before
        GCM_to_goal_distance_before = self.calc_GCM_dist_to_goal()
        num_of_sheep_close_to_goal_before = self.calc_number_of_sheep_near_goal()
        reward = 0

        self.dog.step(action,self.L)
        sheep_dists = self.calc_sheep_dists()
        dog_dists = self.calc_dog_dists()
        sheep_positions = {
            sheep.id:sheep.get_position() for sheep in self.sheep
        }        
        
        for sheep in self.sheep:            
            sheep.step(sheep_positions=sheep_positions, sheep_dists=sheep_dists[sheep.id], dog_position=self.dog.get_position(), dog_dist=dog_dists[sheep.id],L=self.L)

        #obs = self.calc_centroid_based_observation_vector()
        not_herded_sheep_after = self.calc_number_of_sheep_not_hered()
        herded_sheep_after = self.N - not_herded_sheep_after
        GCM_to_goal_distance_after = self.calc_GCM_dist_to_goal()
        num_of_sheep_close_to_goal_after = self.calc_number_of_sheep_near_goal()

        #reward += (herded_sheep_after - herded_sheep_before)
        #reward += herded_sheep_after
        #
        #reward += (num_of_sheep_close_to_goal_after-num_of_sheep_close_to_goal_before)*2
        #reward += num_of_sheep_close_to_goal_after*2
        
        if not_herded_sheep_after < not_herded_sheep_before:
            #print("more sheep herded")
            #reward += 1
            self.number_of_states_of_unherded_sheep = 0
        
        if not_herded_sheep_after > not_herded_sheep_before:            
            self.number_of_states_of_unherded_sheep += 1
            #reward -= 1

        """
        if not_herded_sheep_after == not_herded_sheep_before:
            
            if not_herded_sheep_after == 0:
                #print("all sheep hered")
                #reward += 1
                reward += 0
            else:
                #print("no change in sheep herdness")
                #reward -= 1
                self.number_of_states_of_unherded_sheep += 1
                
        """
        
        if GCM_to_goal_distance_before > GCM_to_goal_distance_after:
            #print("closer")
            #reward += 1
            self.number_of_states_not_closer = 0        

        if GCM_to_goal_distance_before <= GCM_to_goal_distance_after:
            #print("farther")
            #reward -= 1
            self.number_of_states_not_closer += 1               
        
        
        #print("GCM_to_goal_distance_before: " + str(GCM_to_goal_distance_before))
        #print("GCM_to_goal_distance_after: " + str(GCM_to_goal_distance_after))
        #print("not_herded_sheep_before: " + str(not_herded_sheep_before))
        #print("not_herded_sheep_after: " + str(not_herded_sheep_after))
        #print("reward: " + str(reward))
        #print("reward: " + str(reward))
        sheep_dists_from_goal = cdist([sheep_positions[sheep_id] for sheep_id in sheep_positions], [self.goal])
        num_of_sheep_close_to_goal = sum([1 if dist < self.goal_radius else 0 for dist in sheep_dists_from_goal])                
        
        done = False
        
        
        if self.number_of_states_of_unherded_sheep >= 50:
            done = True
            reward -= 1

        if self.number_of_states_not_closer >= 100:
            done = True
            reward -= 1

        if num_of_sheep_close_to_goal == self.N:
            done = True
            print("reward +100000")
            reward += 1

        if self.steps_taken > self.max_steps_taken:
            done = True
            reward -= 1

        

        self.current_reward += reward        
        self.render()
        #print("self.frames[-1].shape: " + str(self.frames[-1].shape))
        return self.frames[-1], reward, done, {}
        #return obs, reward, done, None
        #return None        

    def render(self):
        scaling_factor = 4
        atari_screen_shape = (210, 160, 3)
        
        scaling_factor_h = atari_screen_shape[0]/self.L
        scaling_factor_w = atari_screen_shape[1]/self.L
        #w = self.L*scaling_factor
        
        display = np.zeros(atari_screen_shape, np.uint8)
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        for sheep in self.sheep:
            sheep_x, sheep_y = sheep.get_position()            
            sheep_x = int(sheep_x*scaling_factor_h)
            sheep_y = int(sheep_y*scaling_factor_w)
            sheep_dist_from_centroid = cdist([sheep.get_position()], [GCM])[0][0]
            sheep_dist_from_goal = cdist([sheep.get_position()], [self.goal])[0][0]
            if sheep_dist_from_centroid > max_dist_from_GCM:
                color = (0,0,255)
            else:
                color = (255,255,255)

            if sheep_dist_from_goal <= self.goal_radius:
                color = (0,255,0)
                      
            sheep_min_x = max(sheep_x-scaling_factor, 0)
            sheep_max_x = min(sheep_x+scaling_factor, 210)
            sheep_min_y = max(sheep_y-scaling_factor,0)
            sheep_max_y = min(sheep_y+scaling_factor, 160)

            display[sheep_min_x:sheep_max_x, sheep_min_y:sheep_max_y,:] = color
            
            #display[sheep_x-scaling_factor:sheep_x+scaling_factor, sheep_y-scaling_factor:sheep_y+scaling_factor:,:] = color

        dog_x, dog_y = self.dog.get_position()
        dog_x = int(dog_x*scaling_factor_h)
        dog_y = int(dog_y*scaling_factor_w)                        
        
        color = (255,0,0)                                                                                                    
        dog_min_x = max(dog_x-scaling_factor, 0)
        dog_max_x = min(dog_x+scaling_factor, 210)

        dog_min_y = max(dog_y-scaling_factor,0)
        dog_max_y = min(dog_y+scaling_factor, 160)

        display[dog_min_x:dog_max_x, dog_min_y:dog_max_y,:] = color

        #obs = self.calc_centroid_based_observation_vector()
        #obs = obs.astype(int)        
        #obs *= scaling_factor        
        #
        #display[obs[0]-(scaling_factor):obs[0]+(scaling_factor), obs[1]-(scaling_factor):obs[1]+(scaling_factor),:] = (0,255,0)
        #display[obs[2]-scaling_factor:obs[2]+scaling_factor, obs[3]-scaling_factor:obs[3]+scaling_factor,:] = (0,0,255)

        goal_x, goal_y = self.goal
        goal_x = int(goal_x*scaling_factor_h)
        goal_y = int(goal_y*scaling_factor_w)

        color = (0,255,255)      
        
        goal_min_x = max(goal_x-scaling_factor-5, 0)
        goal_max_x = min(goal_x+scaling_factor+5, 210)
        
        goal_min_y = max(goal_y-scaling_factor-5,0)
        goal_max_y = min(goal_y+scaling_factor+5, 160)

        display[goal_min_x:goal_max_x, goal_min_y:goal_max_y,:] = color
        #display[goal_x-scaling_factor:goal_x+scaling_factor, goal_y-scaling_factor:goal_y+scaling_factor,:] = color
        #img = Image.fromarray(display)
        #draw = ImageDraw.Draw(img)
        ## font = ImageFont.truetype(<font-file>, <font-size>)
        #font = ImageFont.truetype("arial/arial.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        
        #print("self.current_reward: " + str(self.current_reward))
        #print("np.round(self.current_reward,2): " + str(np.round(self.current_reward,2)))
        #draw.text((10, 10),"Current reward: " + str(np.round(self.current_reward,2)) + ", current step: " + str(self.steps_taken),(255,255,255),font=font)
        #print("display.shape: " + str(display.shape))
        #cv.imwrite("atari_screen_sheepherding.png", display)
        #exit()
        #a = 1 / 0
        self.frames.append(display)
        #cv.imshow(title, display)
        #cv.waitKey(1)
        #cv.destroyAllWindows()

    def calc_dog_dists(self):
        sheep_pos = [sheep.get_position() for sheep in self.sheep]
        dog_dists = cdist(sheep_pos, [self.dog.get_position()])
        return {self.sheep[idxi].id: row[0] for idxi, row in enumerate(dog_dists)}

    def calc_sheep_dists(self):        
        sheep_pos = [sheep.get_position() for sheep in self.sheep]        
        sheep_dists = cdist(sheep_pos, sheep_pos)
        return {self.sheep[idxi].id: {self.sheep[idxj].id:dist for idxj, dist in enumerate(row)} for idxi,row in enumerate(sheep_dists)}        

if __name__ == "__main__":
    register(
        id="Sheepherding-v0",
        entry_point="sheepherding:Sheepherding",
        max_episode_steps=300,
        reward_threshold=3000,
    )
    S = gym.make("Sheepherding-v0")
    total_reward = 0
    
    num_of_games_to_play = 300
    total_rewards = []
    
    rmtree("test_games")
    os.makedirs("test_games")
    for i in range(num_of_games_to_play):
        print("i: " + str(i))
        S.reset()
        done = False
        total_reward = 0
        while not done:
            #action = random.sample([0,1,2,3,4,5,6,7],k=1)[0]            
            #_,reward,done,_ = S.step(action)
            _,reward,done,_ = S.step(1)
            _,reward,done,_ = S.step(3)
            #_,reward,done,_ = S.step(0)
            #_,reward,done,_ = S.step(1)
            total_reward += reward
        #S.store_frames_in_mp4("test_games/test.mp4")
        S.store_frames_in_gif("test.gif")
        print("total_reward: " + str(total_reward))
        exit()
        
        total_rewards.append(total_reward)
        #S.store_frames_in_mp4("game.mp4")
        #exit()

    print("np.mean(total_rewards): " + str(np.mean(total_rewards)))
    print("np.std(total_rewards): " + str(np.std(total_rewards)))
    print("np.min(total_rewards): " + str(np.min(total_rewards)))
    print("np.max(total_rewards): " + str(np.max(total_rewards)))


    #for i in range(0,200):
    #    print("")
    #    _,reward, done, _, _ = S.do_action(0,collect=False)        
    #    total_reward += reward
    #    _,reward, done, _, _ = S.do_action(1,collect=False) 
    #    total_reward += reward
    #    #= S.do_action(1)
    #    
    #    #S.render(title=str(i))
    #    print("reward: " + str(reward))
    #    print("total_reward: " + str(total_reward))
    #    
    #S.store_frames_in_mp4()
        
