# Piecewise-Linear Branch and Bound

This repository contains code from the [OVAL](https://www.robots.ox.ac.uk/~oval/) research group for performing Branch and Bound verification on Piecewise-Linear Neural networks,
stemming from a number of research projects.

- [Branch and Bound for Piecewise Linear Neural Network Verification](http://www.jmlr.org/papers/v21/19-468.html)
- [Lagrangian Decomposition for Neural Network Verification](https://arxiv.org/abs/2002.10410)
- [Neural Network Branching for Neural Network Verification](https://arxiv.org/abs/1912.01329) 
  
## Running the code
### Dependencies
The code was implemented assuming to be run under `python3.6`.
We have a dependency on:
* [The Gurobi solver](http://www.gurobi.com/) to solve the LP arising from the
Network linear approximation and the Integer programs for the MIP formulation.
Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
* [Pytorch](http://pytorch.org/) to represent the Neural networks and to use as
  a Tensor library. 

  
### Installation
We assume the user's Python environment is based on Anaconda.

```bash
git clone --recursive https://github.com/oval-group/plnn-bab.git

cd plnn-bab

#Create a conda environment
conda create -n plnn-bab python=3.6
conda activate plnn-bab

# Install gurobipy 
conda config --add channels http://conda.anaconda.org/gurobi
pip install .
#might need
#conda install gurobi

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch 

# Install the code of this repository
python setup.py install
```

### Running the benchmarks

In order to replicate the plots contained in `./plots/` and/or the .csv files in `./mnist_results/csv/` and 
`./cifar_results/csv/`, please execute the following commands:

```bash
# Run the experiments:
# NOTE: it takes more than a day to execute all the benchmarks one after the other
python scripts/bab_tools/bab_runner.py

# Generate plots and .csv files 
python scripts/bab_tools/plot_verification.py
``` 

### Results format

On top of the plots in `./plots/`, plotting the percentage of verified properties over time, 
we provide `.csv` files (in `./mnist_results/csv/` and `./cifar_results/csv/`) with the following format:

`Idx,Eps,SAT,Branches,Time(s)` for the ETH benchmarks (MNIST and CIFAR-10). SAT=True means the network is robust.

`Idx,Eps,prop,BSAT_GNN_prox_100,Branches,Time(s)` for the OVAL benchmarks. SAT=False means the network is robust 
(these properties have been formulated as adversarial vulnerability). The ground truth is False for all.

## Repository structure
* `./plnn/` contains the code for the ReLU Branch and Bound framework.
* `./tools/` contains code to interface the bounds computation classes and the BaB framework. In particular, it 
contains runner code for ReLU-BaB in `tools/bab_tools/bab_runner.py`
* `./scripts/` is a set of bash/python scripts, instrumenting the tools of `./tools`. In particular, 
`scripts/run_batched_verification.py` executes `tools/bab_tools/bab_runner.py`.
* `./models/` contains the trained networks for the challenge, stored in the formats expected by the tool.
* `cifar_properties/` contains the three sets of 100 properties from OVAL's CIFAR-10 verification dataset. 
