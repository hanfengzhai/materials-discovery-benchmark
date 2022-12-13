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

MP_API_KEY="YOUR_KEY"
mpr = MPRester(MP_API_KEY)

def shortlist(long_list, n=5):
    print("First {} of {} items:".format(min(n, 5), len(long_list)))
    for item in long_list[0:n]:
        print(item)

def generate_mat_seq(num_atoms, ele_var): #mat. numbers, mat. elements -> length of mat. cand., list of mat. cand.
    
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
    with MPRester("YOUR_KEY") as mpr:
        mat_found = mpr.get_materials_ids(mat_candidate)#get something like ("Li-Fe-P-O")
        shortlist(mpr.get_materials_ids(mat_candidate))
        # print(mat_found)
    mat_len = len(mat_found)#get length
    
    return mat_len, mat_found#gen_structure


def mat_len_judge(num_atoms, ele_var, mat_opt_pos):
    
    num_atoms, ele_var = int(num_atoms), int(ele_var)
    mat_len_gen = 0
    while mat_len_gen <= 0:
        mat_len_gen, mat_found = generate_mat_seq(num_atoms, ele_var)
    
    mat_opt_pos = int((mat_opt_pos/100)*(mat_len_gen-1))
    mat_id_selection = mat_found[mat_opt_pos]#get material id from MatProj
    with MPRester("YOUR_KEY") as mpr:
        gen_structure = mpr.get_structure_by_material_id(mat_id_selection)#obtain the final structure
    return gen_structure


def material_evaluation(num_atoms, ele_var, mat_opt_pos):#generated_structure
    
    num_atoms, ele_var = int(num_atoms), int(ele_var)
    gen_strctr = mat_len_judge(num_atoms, mat_opt_pos, ele_var)

    ml_model = load_model("logK_MP_2018")#load pretrained model from MEGNet
    structure_input = gen_strctr#generate_mat_seq(num_atoms)#get structure 2b predicted
    print("Input Structure:", structure_input)
    predicted_K = 10 ** ml_model.predict_structure(structure_input).ravel()[0]#get properties from surrogate
    predicted_G = 10 ** ml_model.predict_structure(structure_input).ravel()[1]#get properties from surrogate
    print("Bulk and shear moduli value from MEGNet:", predicted_K, predicted_G)
    
    
    return predicted_K - predicted_G # goal is obtain max bulk w/ min shear mod.


def main():
    num_atoms = random.randint(1,10) 
    ele_var = random.randint(0,100) # initialize optimization with randomization

    mat_len_gen, _ = generate_mat_seq(num_atoms, ele_var)
    
    pbounds = {'num_atoms': (1, 10), 'ele_var': (0, 100), 'mat_opt_pos': (0, 100)} # Set bounds
    
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=1) # GP Upper Confidence Bound

    opt = BayesianOptimization(
        f=material_evaluation,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    
    next_point = opt.suggest(utility)
    target = material_evaluation(**next_point)
    opt.register(
        params=next_point,
        target=target,
    )
    
    opt.maximize(
        init_points=10,
        n_iter=100,
        alpha=1e-3,
    )
    
    for i, res in enumerate(opt.res):
        print("Iteration {}: \n\t{}".format(i, res))
        
    logger = JSONLogger(path="./logs.json") # Save the model
    opt.subscribe(Events.OPTIMIZATION_STEP, logger)

if __name__ == "__main__":
    main()
    print("SUCCESS")


