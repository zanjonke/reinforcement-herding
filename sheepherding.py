import cv2 as cv
import numpy as np
from uuid import uuid4
from scipy.spatial.distance import cdist
from pprint import pprint
from numpy import linalg as LA
import random
#import matplotlib.pyplot as plt
import math

np.random.seed(1)
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

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
        self.step_strength = params["step_strength"]
        self.possible_actions = [0, 1, 2, 3]
        self.step_vectors = {
            0: [0,1],
            1: [1,0],
            2: [0,-1],
            3: [-1,0]
        }

    def set_position(self, position):
        self.position = position

    # move the sheep
    # does not need to return a reward
    def step(self, action, L):
        if not action in self.possible_actions:
            raise Exception("unknown step for dog")        
        move_vector = np.array(self.step_vectors[action])
        new_position = self.position + move_vector*self.step_strength
        if np.logical_or(new_position<0, new_position>L).any():
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
            if np.logical_or(new_position<0, new_position>L).any():

                return 

            self.position = new_position
            self.inertia = H 
            
        else:
            self.graze()
            self.inertia=np.array([0,0])

    def get_position(self):
        return self.position
    

class Sheepherding:
    
    def __init__(self, **params):
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

        self.mode = params["mode"]           # observation vector type -- {'strombom', 'knearest', 'pieslice'}
        self.k = 10
        self.sheep = None
        self.dog = None
        self.goal = params["goal"]
        self.goal_radius = params["goal_radius"]
        self.steps_taken=0
        self.max_steps_taken = params["max_steps_taken"]
        self.action_space = [0,1,2,3]
        self.render_mode = params["render_mode"]
        self.store_video_file_name = params.get("store_video_file_name", "sheepherding_game.mp4")
        self.step_strength = params.get("step_strength", 1)
        self.frames = []
        self.current_reward = 0
        if 'init_values' in params :
            self.init(params['init_values'])
        else :
            self.random_init()

    def store_frames_in_mp4(self):      
        if len(self.frames)==0:
            return
        
        # Define the codec and create VideoWriter object
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        
        out = cv.VideoWriter(self.store_video_file_name,fourcc, 20.0, self.frames[0].shape[0:2])

        for frame in self.frames:            
            out.write(frame)
            
        # Release everything if job is finished        
        out.release()        


    def reset(self):
        self.current_reward = 0
        self.steps_taken=0
        self.random_init()
        self.frames = []
        
        initial_state = self.calc_observation_vector(self.mode)
        state_description = {'L' : self.L,
                             'goal' : self.goal,
                             'dog' : self.dog.get_position(),
                             'sheep' : [sheep.get_position() for sheep in self.sheep]}
        return initial_state, state_description


    # randomly initialize state
    def random_init(self):
        random_x = np.random.uniform(low=0.5,high=0.8, size=self.N)*self.L
        random_y = np.random.uniform(low=0.5,high=0.8, size=self.N)*self.L
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
        self.dog = Dog(starting_position=list(np.random.uniform(low=0.8,high=1.0, size=2)*self.L), ra=self.ra, N=self.N, e=self.e,step_strength=self.step_strength)
        #self.goal = list(np.random.uniform(low=0,high=0.5, size=2)*self.L)

    # initialize state using outside data
    def init(self, init_data):
        self.goal = init_data[0]
        self.dog = Dog(starting_position=np.array(init_data[1]), ra=self.ra,
                       N=self.N, e=self.e, step_strength=self.step_strength)
        self.sheep = [
            Sheep(
                _id=i,
                starting_position=np.array(sheep),
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
            ) for i,sheep in init_data[2]]

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

    def calc_reward(self, collect):
        max_dist_in_map = np.sqrt(self.L**2 + self.L**2)
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        total_reward = 0
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
        total_reward += sum([-1 if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_centroid]) # negative reward
        #max_sheep_dist_from_GCM = max(sheep_dists_from_centroid) 
        #total_reward += sum([-(dist/max_dist_in_map) if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_centroid]) # negative reward
        deformation = total_reward
        dist_to_goal = self.distance(GCM, self.goal)

        sheep_dists_from_goal = cdist(sheep_positions, [self.goal])
        num_of_sheep_close_to_goal = sum([1 if dist < self.goal_radius else 0 for dist in sheep_dists_from_goal])
        reward_from_distances = sum([dist**(-1)/max_dist_in_map for dist in sheep_dists_from_goal])
        #print("reward_from_sheep_close_to_goal: " + str(reward_from_sheep_close_to_goal) + ", N: " + str(self.N))
        #total_reward += num_of_sheep_close_to_goal
        if num_of_sheep_close_to_goal == self.N: # if all sheep are close enough, the game is over
            return reward_from_distances[0] + 1000, True, num_of_sheep_close_to_goal, deformation

        if not collect :
            total_reward = dist_to_goal / (-5) + num_of_sheep_close_to_goal
            #total_reward = self.previous_distance - dist_to_goal
            #total_reward += num_of_sheep_close_to_goal - self.previous_SNG
            #total_reward = -0.5 if abs(total_reward) < 0.01 else total_reward
        else :
            total_reward = deformation
            #total_reward = deformation - self.previous_deformation

        return total_reward, False, num_of_sheep_close_to_goal, deformation

    def calc_observation_vector(self, mode) :
        if mode == 'centroid' :
            return self.calc_centroid_based_observation_vector()
        elif mode == 'knearest' :
            return self.calc_k_nearest_sheep_vector()
        elif mode == 'pieslice' :
            return self.calc_pie_slice_vector()
        else :
            print('Invalid mode selected!')
            return None


    def calc_centroid_based_observation_vector(self):        
        sheep_pos = [sheep.get_position() for sheep in self.sheep]        
        sheep_pos_dict = {sheep.id: sheep.get_position() for sheep in self.sheep}
        sheep_centroid  = np.mean(sheep_pos, axis=0)
        sheep_dists_from_centroid = cdist(sheep_pos, [sheep_centroid])
        sheep_dists_from_centroid_dict = {self.sheep[idxi].id: row[0] for idxi, row in enumerate(sheep_dists_from_centroid)}
        farthest_sheep_from_centroid_id = max(sheep_dists_from_centroid_dict, key=sheep_dists_from_centroid_dict.get)                
        farthest_sheep_from_centroid_position = sheep_pos_dict[farthest_sheep_from_centroid_id]                
        dog_position = self.dog.get_position()
        goal_position = self.goal
        return np.concatenate([sheep_centroid, farthest_sheep_from_centroid_position, dog_position, goal_position],axis=0) / self.L

    def calc_k_nearest_sheep_vector(self):
        dog_position = self.dog.get_position()
        goal_position = self.goal
        sheep_distances = [(self.distance(sheep.get_position(), dog_position), sheep.get_position()) for sheep in self.sheep]
        sheep_distances = sorted(sheep_distances)
        k_nearest_sheep = sheep_distances[:self.k]
        k_nearest_sheep = np.concatenate([sheep[1] for sheep in k_nearest_sheep])
        return np.concatenate([k_nearest_sheep, dog_position, goal_position], axis=0) / self.L

    def calc_pie_slice_vector(self):
        return []

    # move the dog in the wanted position
    # and change the position of the sheep accordingly
    # also return the next state, and reward for this action
    #def step(self, action):
    def do_action(self, action, collect=True):
        """
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra * (self.N ** (2 / 3))

        sheep_positions = [sheep.get_position() for sheep in self.sheep]
        sheep_dists_from_goal = cdist(sheep_positions, [self.goal])
        num_of_sheep_close_to_goal = sum([1 if dist < self.goal_radius else 0 for dist in sheep_dists_from_goal])
        sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
        deformation = sum([-1 if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_centroid])  # negative reward

        self.previous_distance = self.distance(GCM, self.goal)
        self.previous_SNG = num_of_sheep_close_to_goal
        self.previous_deformation = deformation
        """
        self.steps_taken += 1
        self.dog.step(action,self.L)

        sheep_dists = self.calc_sheep_dists()
        dog_dists = self.calc_dog_dists()
        sheep_positions = {
            sheep.id:sheep.get_position() for sheep in self.sheep
        }        

        for sheep in self.sheep:            
            sheep.step(sheep_positions=sheep_positions, sheep_dists=sheep_dists[sheep.id], dog_position=self.dog.get_position(), dog_dist=dog_dists[sheep.id],L=self.L)

        obs = self.calc_observation_vector(self.mode)
        reward, done, sheep_near_goal, deformation = self.calc_reward(collect)

        #print("reward: " + str(reward))
        #exit()
        self.current_reward += reward
        if self.render_mode :
            self.render()

        if done:
            self.store_frames_in_mp4()
            return obs, reward, done, sheep_near_goal, deformation

        if self.steps_taken > self.max_steps_taken:
            self.store_frames_in_mp4()
            return obs, reward, True, sheep_near_goal, deformation

        return obs, reward, False, sheep_near_goal, deformation
        
        #self.dog.strombom_step(sheep_positions, sheep_dists)

    def render(self):
        scaling_factor = 4
        display = np.ones((self.L*scaling_factor, self.L*scaling_factor, 3), np.uint8)
        for sheep in self.sheep:
            sheep_x, sheep_y = sheep.get_position()
            
            sheep_x = int(sheep_x)*scaling_factor
            sheep_y = int(sheep_y)*scaling_factor
            display[sheep_x-scaling_factor:sheep_x+scaling_factor, sheep_y-scaling_factor:sheep_y+scaling_factor:,:] = (255,255,255)

        dog_x, dog_y = self.dog.get_position()
        dog_x = int(dog_x)*scaling_factor
        dog_y = int(dog_y)*scaling_factor
        
        display[dog_x-scaling_factor:dog_x+scaling_factor, dog_y-scaling_factor:dog_y+scaling_factor,:] = (255,0,0)

        obs = self.calc_observation_vector(self.mode)
        obs = obs.astype(int)        
        obs *= scaling_factor        
        

        display[obs[0]-(scaling_factor):obs[0]+(scaling_factor), obs[1]-(scaling_factor):obs[1]+(scaling_factor),:] = (0,255,0)
        display[obs[2]-scaling_factor:obs[2]+scaling_factor, obs[3]-scaling_factor:obs[3]+scaling_factor,:] = (0,0,255)

        goal_x, goal_y = self.goal
        goal_x = int(goal_x)*scaling_factor
        goal_y = int(goal_y)*scaling_factor
        display[goal_x-scaling_factor:goal_x+scaling_factor, goal_y-scaling_factor:goal_y+scaling_factor,:] = (0,255,255)        
        img = Image.fromarray(display)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("Aaargh/Aaargh.ttf", 32)
        # draw.text((x, y),"Sample Text",(r,g,b))
        
        draw.text((0, 0),"Current reward: " + str(np.round(self.current_reward,2)) + ", current step: " + str(self.steps_taken),(255,255,255),font=font)
        self.frames.append(np.array(img))
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
    strombom_typical_values = {
        "N":10,
        "L":150,
        "n":10,
        "rs":15,
        "ra":2,
        "pa":2,
        "c":1.05,
        "ps":1,
        "h":0.5,
        "delta":0.3,
        "p":0.05,
        "e":0.3,
        "delta_s":1.5,
        "goal":[10,10],
        "goal_radius":30,
        "max_steps_taken":500,
        "render_mode": True,
        "mode" : 'centroid'
    }

    # 1 desno
    # 2 dol
    # 3 levo
    # 4 gor
    S = Sheepherding(**strombom_typical_values)
    for i in range(0,200):
        _,reward, done, _, _ = S.do_action(0,collect=False)
        _,reward, done, _, _ = S.do_action(1,collect=False) 
        #= S.do_action(1)

        #S.render(title=str(i))
        print("reward: " + str(reward))
    S.store_frames_in_mp4()
        
