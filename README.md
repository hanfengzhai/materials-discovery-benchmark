# Materials Discovery Benchmark

To run the optimization benchmark cases, please go to `inverse_opt/`

This materials discovery framework utilized the pre-trained MEGNet models, which have been summarized (adapted from the original repo) as follows:

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
