from mealpy.swarm_based.ACOR import OriginalACOR

import numpy as np
import random
from pymatgen.core import Composition
from matminer.featurizers.composition.element import ElementFraction
from mp_api.client import MPRester
from megnet.utils.models import load_model
from pymatgen.core import Structure, Lattice

from ctypes import *
import os


MP_API_KEY="YOUR_API_KEY"
mpr = MPRester(MP_API_KEY)

def shortlist(long_list, n=5):
    print("First {} of {} items:".format(min(n, 5), len(long_list)))
    for item in long_list[0:n]:
        print(item)

def generate_mat_seq(X): #mat. numbers, mat. elements -> length of mat. cand., list of mat. cand.
    
    num_atoms, ele_var = X[0], X[1]
    
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
        shortlist(mpr.get_materials_ids(mat_candidate))
        # print(mat_found)
    mat_len = len(mat_found)#get length
    
    return mat_len, mat_found#gen_structure


def mat_len_judge(X):
    
    num_atoms, ele_var, mat_opt_pos = X[0], X[1], X[2]
    
    num_atoms, ele_var = int(num_atoms), int(ele_var)
    mat_len_gen = 0
    while mat_len_gen <= 0:
        X_temp = [num_atoms, ele_var]
        mat_len_gen, mat_found = generate_mat_seq(X_temp)#num_atoms, ele_var
    
    mat_opt_pos = int((mat_opt_pos/100)*(mat_len_gen-1))
    mat_id_selection = mat_found[mat_opt_pos]#get material id from MatProj
    with MPRester("YOUR_API_KEY") as mpr:
        gen_structure = mpr.get_structure_by_material_id(mat_id_selection)#obtain the final structure
    return gen_structure


def material_evaluation(X):#generated_structure
    
    num_atoms, ele_var, mat_opt_pos = int(X[0]), int(X[1]), int(X[2])
    
    X = [num_atoms, ele_var, mat_opt_pos]
    gen_strctr = mat_len_judge(X)
    print("Design Parameters:", X)
    model_K = load_model("logK_MP_2019")#load pretrained model from MEGNet
    model_Efermi = load_model("Efermi_MP_2019")#load pretrained model from MEGNet
    structure_input = gen_strctr#generate_mat_seq(num_atoms)#get structure 2b predicted
    print("Input Structure:", structure_input)
    predicted_K = 10 ** model_K.predict_structure(structure_input).ravel()[0]#get properties from surrogate
    predicted_Efermi = model_Efermi.predict_structure(structure_input).ravel()[0]#get properties from surrogate
    print("Bulk modulus & Efermi predictions from MEGNet:", predicted_K, predicted_Efermi)
    print("====================================")
    
    return - predicted_K + predicted_Efermi # goal is obtain max bulk w/ min shear mod.

def testfunc(X):
    print("yes!")
    return np.mean(X)

def main():
    num_atoms = random.randint(1,10) 
    ele_var = random.randint(0,100) # initialize optimization with randomization

    X_init = [num_atoms, ele_var, 0]
    mat_len_gen, _ = generate_mat_seq(X_init)
    
    problem_dict1 = {
        "fit_func": material_evaluation,
        "lb": [1, 0, 0],
        "ub": [4, 100, 100],
        "minmax": "min",
    }

    epoch = 60
    pop_size = 10
    
    sample_count = 5
    intent_factor = 0.5
    zeta = 1.0
    model = OriginalACOR(epoch, pop_size, sample_count, intent_factor, zeta)  
    
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")

if __name__ == "__main__":
    main()
    print("SUCCESS")
