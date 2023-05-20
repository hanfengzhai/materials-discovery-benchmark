# Materials Discovery Benchmark


**This materials discovery framework utilized the pretraianed MEGNet models, which has summarized (adapted from the original repo) as follows:**



***




<a name="megnet-framework"></a>

# MEGNet framework

Copyright (c) 2015, Regents of the University of California All rights reserved.


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
