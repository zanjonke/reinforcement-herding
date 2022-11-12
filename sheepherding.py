import cv2 as cv
import numpy as np
from uuid import uuid4
from scipy.spatial.distance import cdist
from pprint import pprint
from numpy import linalg as LA

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


    def set_position(self, position):
        self.position = position

    # move the sheep
    # does not need to return a reward
    def step(self, action):
        pass
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
        #print("calc_LCM: ")
        closest_sheep = sorted(sheep_dists, key=sheep_dists.get)[1:self.n+1]
        #print("closest_sheep: " + str(closest_sheep))
        closest_sheep_positions = [sheep_positions[sheep_id] for sheep_id in closest_sheep]
        #print("closest_sheep_positions: " + str(closest_sheep_positions))
        return np.mean(closest_sheep_positions, axis=0)

    def calc_repulsion_force_vector_from_too_close_neighborhood(self,sheep_positions,sheep_dists):
        too_close_sheep = list(filter(lambda sheep_id: sheep_dists[sheep_id]<self.ra and sheep_id != self.id, sheep_dists))
        if len(too_close_sheep)==0:
            return np.array([0,0])    
        
        Ra = [(self.position-sheep_positions[sheep_id])/(LA.norm(self.position-sheep_positions[sheep_id])) for sheep_id in too_close_sheep]
        print("PRE Ra: " + str(Ra))
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
    def step(self,sheep_positions, sheep_dists, dog_position, dog_dist):                                                        
        
        if dog_dist < self.rs:
            print("UPDATING")
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
            
            self.position += self.delta*H
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
        
        self.random_init()

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

    # move the dog in the wanted position
    # and change the position of the sheep accordingly
    # also return the next state, and reward for this action
    def step(self, action):
        sheep_dists = self.calc_sheep_dists()
        dog_dists = self.calc_dog_dists()
        sheep_positions = {
            sheep.id:sheep.get_position() for sheep in self.sheep
        }
        for sheep in self.sheep:            
            sheep.step(sheep_positions=sheep_positions, sheep_dists=sheep_dists[sheep.id], dog_position=self.dog.get_position(), dog_dist=dog_dists[sheep.id])

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
    typical_values = {
        "N":60,
        "L":150,
        "n":10,
        "rs":120,
        "ra":2,
        "pa":2,
        "c":1.05,
        "ps":1,
        "h":0.5,
        "delta":0.3,
        "p":0.05,
        "e":0.3,
        "delta_s":1.5
    }
    S = Sheepherding(**typical_values)
    for i in range(0,200):
        S.step(None)
        S.render(title=str(i))
        print("i: " + str(i))
