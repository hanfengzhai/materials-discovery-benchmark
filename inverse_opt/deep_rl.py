from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

from pymatgen.core import Composition
from matminer.featurizers.composition.element import ElementFraction
from mp_api.client import MPRester
from megnet.utils.models import load_model
from pymatgen.core import Structure, Lattice

from geneticalgorithm import geneticalgorithm as ga
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours
from ctypes import *
import os
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import matplotlib.pyplot as plt
from mpi4py import MPI
from megnet.utils.data import get_graphs_within_cutoff # avoid bulk modulus errors

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from mpi4py import MPI


MP_API_KEY="YOUR_API_KEY"
mpr = MPRester(MP_API_KEY)

class MatEnv(Env):
    def __init__(self):#, num_atoms, ele_var, mat_opt_pos
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state = 200 + random.randint(-100,100)
        # Set shower length
        self.shower_length = 5
        self.num_atoms = int(random.randint(1,4))
        self.ele_var = int(random.randint(0,100))
        self.mat_opt_pos = int(random.randint(0,100))
    def get_params(self):
        return num_atoms, ele_var, mat_opt_pos

    def shortlist(self, long_list, n=5):
        print("First {} of {} items:".format(min(n, 5), len(long_list)))
        for item in long_list[0:n]:
            print(item)

    def generate_mat_seq(self, num_atoms, ele_var): #mat. numbers, mat. elements -> length of mat. cand., list of mat. cand.
        
        num_atoms = int(num_atoms)
        element_numlabel = np.ones(num_atoms) #generate an array to store atoms
        ef = ElementFraction()
        element_fraction_labels = ef.feature_labels()
        
        for i in range(num_atoms):
            ele_var = int(((np.array((ele_var/100)) + np.random.rand(1))/2)*102)
            element_numlabel[i] = ele_var

        element_symbols = []
        for j in element_numlabel:
            symbols_vars = element_fraction_labels[int(j)]#get symbolized element variables
            element_symbols.append(symbols_vars)#append vars. to a list

        candidate_elements = [str(k) for k in element_symbols]#loop thru candidate materials elements
        mat_candidate = '-'.join(candidate_elements)#get materials cobination in a string form w/ '-'
        print("Materials Basis:", mat_candidate)

        # mat_found = []
        with MPRester("YOUR_API_KEY") as mpr:
            mat_found = mpr.get_materials_ids(mat_candidate)#get something like ("Li-Fe-P-O")
            self.shortlist(mpr.get_materials_ids(mat_candidate))
            # print(mat_found)
        mat_len = len(mat_found)#get length
        
        return mat_len, mat_found#gen_structure


    def mat_len_judge(self, num_atoms, ele_var, mat_opt_pos):
        
        # num_atoms, ele_var = int(num_atoms), int(ele_var)
        mat_len_gen = 0
        while mat_len_gen <= 0:
            mat_len_gen, mat_found = self.generate_mat_seq(num_atoms, ele_var)
        
        mat_opt_pos = int((mat_opt_pos/100)*(mat_len_gen-1))
        mat_id_selection = mat_found[mat_opt_pos]#get material id from MatProj
        with MPRester("YOUR_API_KEY") as mpr:
            gen_structure = mpr.get_structure_by_material_id(mat_id_selection)#obtain the final structure
        return gen_structure

    def material_evaluation(self, action):#generated_structure
        num_atoms, ele_var, mat_opt_pos = self.get_params()
        print("action:",action)
        if action == 0:
            num_atoms = random.randint(1,4); num_atoms = int(num_atoms)#generate new set
        elif action == 1:
            ele_var = random.randint(0,100); ele_var = int(ele_var)
        elif action == 2:
            mat_opt_pos = random.randint(0,100); mat_opt_pos = int(mat_opt_pos)
        
        # num_atoms, ele_var = int(num_atoms), int(ele_var)
        gen_strctr = self.mat_len_judge(num_atoms, ele_var, mat_opt_pos)

        model_K = load_model("logK_MP_2019")#load pretrained model from MEGNet
        model_Efermi = load_model("Efermi_MP_2019")#load pretrained model from MEGNet
        structure_input = gen_strctr#generate_mat_seq(num_atoms)#get structure 2b predicted
        print("Input Structure:", structure_input)
        predicted_K = 10 ** model_K.predict_structure(structure_input).ravel()[0]#get properties from surrogate
        predicted_Efermi = model_Efermi.predict_structure(structure_input).ravel()[0]#get properties from surrogate
        print("Bulk modulus & Efermi predictions from MEGNet:", predicted_K, predicted_Efermi)
        print("====================================")
        
        return predicted_K - predicted_Efermi # goal is obtain max bulk w/ min shear mod.

    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        #num_atoms, ele_var, mat_opt_pos = np.array(num_atoms), np.array(ele_var), np.array(mat_opt_pos)
        
        self.state = self.material_evaluation(action)
        
        # ***self.state += action - 1 
        # Reduce shower length by 1 second
        
        
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >=350: 
            reward = 10 
        elif self.state >=250 and self.state <=350: 
            reward = 5
        elif self.state >=150 and self.state <=250: 
            reward = 1
        elif self.state >=50 and self.state <=150:    
            reward = -5
        elif self.state <=50:
            reward = -10
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = 200 + random.randint(-100,100)
        # Reset shower time
        self.shower_length = 5
        return self.state
    

env = MatEnv()

print("obs. samples: ",env.observation_space.sample())

episodes = 30
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        num_atoms = int(random.randint(1,4))
        ele_var = int(random.randint(0,100))
        mat_opt_pos = int(random.randint(0,100))
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))


states = env.observation_space.shape
actions = env.action_space.n; print("actions:",actions)

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=states))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(states, actions)

# print(model.summary())

print(model.summary())

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn

del model 
model = build_model(states, actions)

dqn = build_agent(model, actions)


dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=60, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=20, visualize=False)
print(np.mean(scores.history['episode_reward']))

# _ = dqn.test(env, nb_episodes=10, visualize=False)
print("SUCCESS")


