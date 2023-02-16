[![License](https://img.shields.io/github/license/materialsvirtuallab/megnet)]()
[![Build Status](https://travis-ci.org/materialsvirtuallab/megnet.svg?branch=master)](https://travis-ci.org/materialsvirtuallab/megnet)
[![Coverage Status](https://coveralls.io/repos/github/materialsvirtuallab/megnet/badge.svg?branch=master)](https://coveralls.io/github/materialsvirtuallab/megnet?branch=master&service=github)
[![Downloads](https://pepy.tech/badge/megnet)](https://pepy.tech/project/megnet)
[![Linting](https://github.com/materialsvirtuallab/megnet/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/megnet/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/megnet/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/megnet/workflows/Testing%20-%20main/badge.svg)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/materialsvirtuallab/megnet/master)


**This materials discovery framework utilized the MEGNet framework, which has summarized (adapted from the original repo) as follows:**



Copyright (c) 2015, Regents of the University of California All rights reserved.

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
