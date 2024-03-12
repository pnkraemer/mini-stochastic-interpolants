# Stochastic interpolants


<p align="center">
    <img src="https://github.com/pnkraemer/mini-stochastic-interpolants/blob/main/name_sample_animation.gif" width="400" height="225" />
</p>

This repository contains a minimal implementation of some concepts related to stochastic interpolants in JAX, based on [this paper](https://arxiv.org/abs/2303.08797) by Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.


**Disclaimer:**
This implementation is meant to be didactic.
For a more functional version (in Pytorch), see the repository published by the authors of the paper [here](https://github.com/malbergo/stochastic-interpolants).



## Installation

Before installing this project, 
and [after creating & activating your virtual environment](https://realpython.com/python-virtual-environments-a-primer/), 
you must install JAX yourself because CPU and GPU backends require different installation commands.
See [here](https://jax.readthedocs.io/en/latest/installation.html) for instructions.
For the small examples, `pip install jax[cpu]` will suffice. 
For the bigger demos, a GPU is helpful.


Then, move to the root of the directory and run
```
pip install .
```
This command installs all requirements (Flax, Optax, etc.).

Then, find the content as
```
from stochint import *
```

## Demonstrations

Find the demos in `demos/`.


## Acknowledgements

Thanks to Paul Jeha (@pablo2909) for teaching us how to write a name with 2d samples.

