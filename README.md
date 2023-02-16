[![License](https://img.shields.io/github/license/materialsvirtuallab/megnet)]()
[![Build Status](https://travis-ci.org/materialsvirtuallab/megnet.svg?branch=master)](https://travis-ci.org/materialsvirtuallab/megnet)
[![Coverage Status](https://coveralls.io/repos/github/materialsvirtuallab/megnet/badge.svg?branch=master)](https://coveralls.io/github/materialsvirtuallab/megnet?branch=master&service=github)
[![Downloads](https://pepy.tech/badge/megnet)](https://pepy.tech/project/megnet)
[![Linting](https://github.com/materialsvirtuallab/megnet/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/megnet/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/megnet/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/megnet/workflows/Testing%20-%20main/badge.svg)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/materialsvirtuallab/megnet/master)


**This materials discovery framework utilized the pretraianed MEGNet models, which has summarized (adapted from the original repo) as follows:**



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

