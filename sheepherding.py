import cv2 as cv
import numpy as np
from uuid import uuid4
from scipy.spatial.distance import cdist
from pprint import pprint
from numpy import linalg as LA
import random
np.random.seed(1)
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
        self.step_strength = 1
        self.possible_actions = [0, 1, 2, 3]
        # 0 is stay ?
        # 1 is move up
        # 2 is move right
        # 3 is move down
        # 4 is move left
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
        
        self.sheep = None
        self.dog = None
        self.goal = params["goal"]
        self.goal_radius = params["goal_radius"]
        self.random_init()
        self.steps_taken=0
        self.max_steps_taken = params["max_steps_taken"]
        self.action_space = [0,1,2,3]
        self.render_mode = params["render_mode"]

    def reset(self):
        self.steps_taken=0
        self.random_init()

        initial_state = self.calc_centroid_based_observation_vector()
        return initial_state


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
        self.dog = Dog(starting_position=list(np.random.uniform(low=0,high=0.5, size=2)*self.L), ra=self.ra, N=self.N, e=self.e)

    def calc_GCM(self):        
        return np.mean([sheep.get_position() for sheep in self.sheep],axis=0)

    def calc_reward(self):
        sheep_positions = [sheep.get_position() for sheep in self.sheep]    
        total_reward = 0
        GCM = self.calc_GCM()
        max_dist_from_GCM = self.ra*(self.N**(2/3))
        sheep_dists_from_centroid = cdist(sheep_positions, [GCM])
        total_reward += sum([-1 if dist > max_dist_from_GCM else 0 for dist in sheep_dists_from_centroid]) # negative reward

        sheep_dists_from_goal = cdist(sheep_positions, [self.goal])
        reward_from_sheep_close_to_goal = sum([1 if dist < self.goal_radius else 0 for dist in sheep_dists_from_goal])
        #print("reward_from_sheep_close_to_goal: " + str(reward_from_sheep_close_to_goal) + ", N: " + str(self.N))
        total_reward += reward_from_sheep_close_to_goal
        if reward_from_sheep_close_to_goal == self.N: # if all sheep are close enough, the game is over
            return reward_from_sheep_close_to_goal, True, reward_from_sheep_close_to_goal

        return total_reward, False, reward_from_sheep_close_to_goal

    def calc_centroid_based_observation_vector(self):        
        sheep_pos = [sheep.get_position() for sheep in self.sheep]        
        sheep_pos_dict = {sheep.id: sheep.get_position() for sheep in self.sheep}
        sheep_centroid  = np.mean(sheep_pos,axis=0)        
        sheep_dists_from_centroid = cdist(sheep_pos, [sheep_centroid])
        sheep_dists_from_centroid_dict = {self.sheep[idxi].id: row[0] for idxi, row in enumerate(sheep_dists_from_centroid)}
        farthest_sheep_from_centroid_id = max(sheep_dists_from_centroid_dict, key=sheep_dists_from_centroid_dict.get)                
        farthest_sheep_from_centroid_position = sheep_pos_dict[farthest_sheep_from_centroid_id]                
        dog_position = self.dog.get_position()
        goal_position = self.goal
        return np.concatenate([sheep_centroid, farthest_sheep_from_centroid_position, dog_position, goal_position],axis=0)

    # move the dog in the wanted position
    # and change the position of the sheep accordingly
    # also return the next state, and reward for this action
    #def step(self, action):
    def do_action(self, action):

        self.steps_taken += 1
        self.dog.step(action,self.L)

        sheep_dists = self.calc_sheep_dists()
        dog_dists = self.calc_dog_dists()
        sheep_positions = {
            sheep.id:sheep.get_position() for sheep in self.sheep
        }        

        for sheep in self.sheep:            
            sheep.step(sheep_positions=sheep_positions, sheep_dists=sheep_dists[sheep.id], dog_position=self.dog.get_position(), dog_dist=dog_dists[sheep.id],L=self.L)

        obs = self.calc_centroid_based_observation_vector()
        reward, done, sheep_near_goal = self.calc_reward()

        if self.render_mode :
            self.render(str(self.steps_taken))

        if done:
            return obs, reward, done, sheep_near_goal

        if self.steps_taken > self.max_steps_taken:
            return obs, reward, True, sheep_near_goal

        return obs, reward, False, sheep_near_goal
        
        #self.dog.strombom_step(sheep_positions, sheep_dists)

    def render(self, title):
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

        obs = self.calc_centroid_based_observation_vector()
        obs = obs.astype(int)        
        obs *= scaling_factor        
        

        display[obs[0]-(scaling_factor*4):obs[0]+(scaling_factor*4), obs[1]-(scaling_factor*4):obs[1]+(scaling_factor*4),:] = (0,255,0)
        display[obs[2]-scaling_factor:obs[2]+scaling_factor, obs[3]-scaling_factor:obs[3]+scaling_factor,:] = (0,0,255)

        goal_x, goal_y = self.goal
        goal_x = int(goal_x)*scaling_factor
        goal_y = int(goal_y)*scaling_factor
        display[goal_x-scaling_factor:goal_x+scaling_factor, goal_y-scaling_factor:goal_y+scaling_factor,:] = (0,255,255)

        cv.imshow(title, display)
        cv.waitKey()
        cv.destroyAllWindows()

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
        "N":50,
        "L":150,
        "n":10,
        "rs":50,
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
        "render_mode": True
    }

    # 1 desno
    # 2 dol
    # 3 levo
    # 4 gor
    S = Sheepherding(**strombom_typical_values)
    for i in range(0,200):
        S.do_action(0)
        _,reward, done, _ = S.do_action(1)
        
        #S.render(title=str(i))
        print("reward: " + str(reward))
        
