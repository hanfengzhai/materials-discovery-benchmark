[![License](https://img.shields.io/github/license/materialsvirtuallab/megnet)]()
[![Build Status](https://travis-ci.org/materialsvirtuallab/megnet.svg?branch=master)](https://travis-ci.org/materialsvirtuallab/megnet)
[![Coverage Status](https://coveralls.io/repos/github/materialsvirtuallab/megnet/badge.svg?branch=master)](https://coveralls.io/github/materialsvirtuallab/megnet?branch=master&service=github)
[![Downloads](https://pepy.tech/badge/megnet)](https://pepy.tech/project/megnet)
[![Linting](https://github.com/materialsvirtuallab/megnet/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/megnet/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/megnet/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/megnet/workflows/Testing%20-%20main/badge.svg)


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/materialsvirtuallab/megnet/master)

This materials discovery framework utilized the MEGNet framework, which has summarized (adapted from the original repo) as follows:


<a name="megnet-framework"></a>
# MEGNet framework

The MatErials Graph Network (MEGNet) is an implementation of DeepMind's graph
networks[1] for universal machine learning in materials science. We have
demonstrated its success in achieving very low prediction errors in a broad
array of properties in both molecules and crystals (see
["Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"](https://doi.org/10.1021/acs.chemmater.9b01294)[2]). New releases have included our recent work on multi-fidelity materials property modeling (See ["Learning properties of ordered and disordered materials from multi-fidelity data"](https://www.nature.com/articles/s43588-020-00002-x)[3]).

Briefly, Figure 1 shows the sequential update steps of the graph network,
whereby bonds, atoms, and global state attributes are updated using information
from each other, generating an output graph.

![GraphModel diagram](resources/model_diagram_small.jpg)
<div align='center'><strong>Figure 1. The graph network update function.</strong></div>

Figure 2 shows the overall schematic of the MEGNet. Each graph network module
is preceded by two multi-layer perceptrons (known as Dense layers in Keras
terminology), constituting a MEGNet block. Multiple MEGNet blocks can be
stacked, allowing for information flow across greater spatial distances. The
number of blocks required depend on the range of interactions necessary to
predict a target property. In the final step, a `set2set` is used to map the
output to a scalar/vector property.

![GraphModel architecture](resources/model_arch_small.jpg)
<div align='center'><strong>Figure 2. Schematic of MatErials Graph Network.</strong></div>


* Materials Project data:
    - Formation energy from the elements
    - Band gap
    - Log 10 of Bulk Modulus (K)
    - Log 10 of Shear Modulus (G)

The MAEs on the various models are given below:

### Performance of QM9 MEGNet-Simple models

| Property | Units      | MAE   |
|----------|------------|-------|
| HOMO     | eV         | 0.043 |
| LUMO     | eV         | 0.044 |
| Gap      | eV         | 0.066 |
| ZPVE     | meV        | 1.43  |
| µ        | Debye      | 0.05  |
| α        | Bohr^3     | 0.081 |
| \<R2\>   | Bohr^2     | 0.302 |
| U0       | eV         | 0.012 |
| U        | eV         | 0.013 |
| H        | eV         | 0.012 |
| G        | eV         | 0.012 |
| Cv       | cal/(molK) | 0.029 |
| ω1       | cm^-1      | 1.18  |

### Performance of MP-2018.6.1

| Property | Units      | MAE   |
|----------|------------|-------|
| Ef       | eV/atom    | 0.028 |
| Eg       | eV         | 0.33  |
| K_VRH    | log10(GPa) | 0.050 |
| G_VRH    | log10(GPa) | 0.079 |

### Performance of MP-2019.4.1

| Property | Units      | MAE   |
|----------|------------|-------|
| Ef       | eV/atom    | 0.026 |
| Efermi   | eV         | 0.288 |

New models will be added as they are developed in the [mvl_models](mvl_models)
folder. Each folder contains a summary of model details and benchmarks. For
the initial models and bencharmks comparison to previous models,
please refer to ["Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"](https://doi.org/10.1021/acs.chemmater.9b01294)[2].

Below is an example of crystal model usage:

```python
from megnet.utils.models import load_model
from pymatgen.core import Structure, Lattice

# load a model in megnet.utils.models.AVAILABLE_MODELS
model = load_model("logK_MP_2018")

# We can construct a structure using pymatgen
structure = Structure(Lattice.cubic(3.167),
            ['Mo', 'Mo'], [[0, 0, 0], [0.5, 0.5, 0.5]])


# Use the model to predict bulk modulus K. Note that the model is trained on
# log10 K. So a conversion is necessary.
predicted_K = 10 ** model.predict_structure(structure).ravel()[0]
print(f'The predicted K for {structure.composition.reduced_formula} is {predicted_K:.0f} GPa.')
```
A full example is in [notebooks/crystal_example.ipynb](notebooks/crystal_example.ipynb).

For molecular models, we have an example in
[notebooks/qm9_pretrained.ipynb](notebooks/qm9_pretrained.ipynb).
We support prediction directly from a pymatgen molecule object. With a few more
lines of code, the model can predict from `SMILES` representation of molecules,
as shown in the example. It is also straightforward to load a `xyz` molecule
file with pymatgen and predict the properties using the models. However, the
users are generally not advised to use the `qm9` molecule models for other
molecules outside the `qm9` datasets, since the training data coverage is
limited.

Below is an example of predicting the "HOMO" of a smiles representation

```python
from megnet.utils.molecule import get_pmg_mol_from_smiles
from megnet.models import MEGNetModel

# same model API for molecule and crystals, you can also use the load_model method
# as in previous example
model = MEGNetModel.from_file('mvl_models/qm9-2018.6.1/HOMO.hdf5')
# Need to convert SMILES into pymatgen Molecule
mol = get_pmg_mol_from_smiles("C")
model.predict_structure(mol)
```

## Training a new MEGNetModel from structures

For users who wish to build a new model from a set of crystal structures with
corresponding properties, there is a convenient `MEGNetModel` class for setting
up and training the model. By default, the number of MEGNet blocks is 3 and the
atomic number Z is used as the only node feature (with embedding).

```python
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np

nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)

# Model training
# Here, `structures` is a list of pymatgen Structure objects.
# `targets` is a corresponding list of properties.
model.train(structures, targets, epochs=10)

# Predict the property of a new structure
pred_target = model.predict_structure(new_structure)
```
Note that for realistic models, the `nfeat_bond` can be set to 100 and `epochs` can be 1000.
In some cases, some structures within the training pool may not be valid (containing isolated atoms),
then one needs to use `train_from_graphs` method by training only on the valid graphs.

Following the previous example,
```python
model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
graphs_valid = []
targets_valid = []
structures_invalid = []
for s, p in zip(structures, targets):
    try:
        graph = model.graph_converter.convert(s)
        graphs_valid.append(graph)
        targets_valid.append(p)
    except:
        structures_invalid.append(s)

# train the model using valid graphs and targets
model.train_from_graphs(graphs_valid, targets_valid)
```
For model details and benchmarks, please refer to ["Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"](https://doi.org/10.1021/acs.chemmater.9b01294)[2]


### Training multi-fidelity graph networks

Please see the folder `multifidelity` for specific examples.

### Pre-trained elemental embeddings

A key finding of our work is that element embeddings from trained formation
energy models encode useful chemical information that can be transferred
learned to develop models with smaller datasets (e.g. elastic constants,
band gaps), with better converegence and lower errors. These embeddings are
also potentially useful in developing other ML models and applications. These
embeddings have been made available via the following code:

```python
from megnet.data.crystal import get_elemental_embeddings

el_embeddings = get_elemental_embeddings()
```

An example of transfer learning using the elemental embedding from formation
energy to other models, please check [notebooks/transfer_learning.ipynb](notebooks/transfer_learning.ipynb).



<a name="references"></a>
# References

1. Battaglia, P. W.; Hamrick, J. B.; Bapst, V.; Sanchez-Gonzalez, A.;
   Zambaldi, V.; Malinowski, M.; Tacchetti, A.; Raposo, D.; Santoro, A.;
   Faulkner, R.; et al. Relational inductive biases, deep learning, and graph
   networks. 2018, 1–38. [arXiv:1806.01261](https://arxiv.org/abs/1806.01261)
2. Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a
   Universal Machine Learning Framework for Molecules and Crystals. Chemistry
   of Materials 2019, 31(9), 3564-3572.
   [doi:10.1021/acs.chemmater.9b01294](https://doi.org/10.1021/acs.chemmater.9b01294)
3. Chen, C.; Zuo, Y.; Ye, W.; Li, X.G.; Ong, S. P. Learning properties of ordered and
   disordered materials from multi-fidelity data. Nature Computational Science 2021,
   1, 46–53 [doi:10.1038/s43588-020-00002-x](https://www.nature.com/articles/s43588-020-00002-x).
4. Vinyals, O.; Bengio, S.; Kudlur, M. Order Matters: Sequence to sequence for
   sets. 2015, arXiv preprint. [arXiv:1511.06391](https://arxiv.org/abs/1511.06391)
5. https://figshare.com/articles/Graphs_of_materials_project/7451351
6. Ong, S. P.; Cholia, S.; Jain, A.; Brafman, M.; Gunter, D.; Ceder, G.;
   Persson, K. A. The Materials Application Programming Interface (API): A
   simple, flexible and efficient API for materials data based on
   REpresentational State Transfer (REST) principles. Comput. Mater. Sci. 2015,
   97, 209–215 DOI: [10.1016/j.commatsci.2014.10.037](http://dx.doi.org/10.1016/j.commatsci.2014.10.037).
7. Faber, F. A.; Hutchison, L.; Huang, B.; Gilmer, J.; Schoenholz, S. S.;
   Dahl, G. E.; Vinyals, O.; Kearnes, S.; Riley, P. F.; von Lilienfeld, O. A.
   Prediction errors of molecular machine learning models lower than hybrid DFT
   error. Journal of Chemical Theory and Computation 2017, 13, 5255–5264.
   DOI: [10.1021/acs.jctc.7b00577](http://dx.doi.org/10.1021/acs.jctc.7b00577)
