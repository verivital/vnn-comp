# Piecewise-Linear Branch and Bound

This repository contains code for performing Branch and Bound verification on Piecewise-Linear Neural network,
stemming from a number of research projects.

## Neural Network bounds
The repository provides code for algorithms to compute output bounds for ReLU-based neural networks (and, 
more generally, piecewise-linear networks, which can be transformed into equivalent ReLUs):
- `LinearizedNetwork` in `plnn/network_linear_approximation.py` represents the [PLANET](https://github.com/progirep/planet) relaxation of the network in Gurobi 
and uses the commercial solver to compute the model's output bounds.
- `SaddleLP` in `plnn/proxlp_solver/solver.py` implements the dual iterative algorithms presented in 
["Lagrangian Decomposition for Neural Network Verification"](https://arxiv.org/abs/2002.10410) in PyTorch, based on the Lagrangian Decomposition of the activation's 
convex relaxations.
- `DJRelaxationLP` in `plnn/proxlp_solver/dj_relaxation.py` implements the Lagrangian relaxation-based dual iterative algorithm presented in 
"[A Dual Approach to Scalable Verification of Deep Networks](https://arxiv.org/abs/1803.06567)" in PyTorch.

These classes offer two main interfaces (see `tools/cifar_runner.py` and `tools/cifar_bound_comparison.py` for detailed 
usage, including algorithm parametrization):
- Given some pre-computed intermediate bounds, compute the bounds on the neural network output: 
call `build_model_using_bounds`, then `compute_lower_bound`.
- Compute bounds for activations of all network layers, one after the other (each layer's computation will use the 
bounds computed for the previous one): `define_linear_approximation`.

The computed neural network bounds can be employed in two different ways: alone, to perform incomplete 
verification; as the bounding part of branch and bound (to perform complete verification).

## Implementation details

The dual iterative algorithms (`SaddleLP`, `DJRelaxationLP`) **batch** their computations on two different dimensions:
the first is the number of different domains to solve at once (passed as a batch of input domains, and/or intermediate 
bounds). The second batch dimension is over each layer output lower/upper bounds in order to compute them in parallel.

## Branch and Bound design

The ReLU Branch and Bound framework (`plnn/branch_and_bound/relu_branch_and_bound`) employs ReLU splitting (input domain
 splitting is not supported at the moment) and chooses the ReLU to switch on according to the KW heuristic introduced in
 [Branch and Bound for Piecewise Linear Neural Network Verification](http://www.jmlr.org/papers/v21/19-468.html) 
 (BaBSR). 
In case the dual iterative algorithms are employed for the bounding, a number of BaB sub-problems (nodes) is solved in 
parallel at once as a PyTorch batch. 

## Repository structure
* `./plnn/` contains the code for the ReLU Branch and Bound framework (`plnn/branch_and_bound/relu_branch_and_bound`), 
along with utilities and classes to compute bounds on neural network outputs mentioned above.
* `./tools/` contains code to interface the bounds computation classes and the BaB framework. In particular, it 
contains runner code for ReLU-BaB in `tools/bab_tools/bab_runner.py`
* `./scripts/` is a set of bash/python scripts, instrumenting the tools of `./tools`. In particular, 
`scripts/run_batched_verification.py` executes `tools/bab_tools/bab_runner.py`.
* `./models/` contains the trained networks for CIFAR10 employed in 
[Neural Network Branching for Neural Network Verification](https://arxiv.org/abs/1912.01329).
* `batch_verification_results/` contains pickled results of BaB runs on CIFAR-10. 
  
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
