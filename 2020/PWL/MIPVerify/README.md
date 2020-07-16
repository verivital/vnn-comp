# MIPVerify
This folder contains code to reproduce the results for the [MIPVerify](https://github.com/vtjeng/MIPVerify.jl) tool on the [benchmark for piecewise linear activations](https://github.com/verivital/vnn-comp/issues/2).

## Manual Installation
  1. Install Julia, Gurobi, a Gurobi License, and relevant packages. [Installation instructions](https://vtjeng.github.io/MIPVerify.jl/v0.2.2/#Installation-1).

## Reproducing Results
### System Specifications
**CPU**: Intel(R) Core(TM) i5-4300U CPU @ 1.90GHz (via `cat /proc/cpuinfo | grep "model name"`)
**RAM**: 8 GB, DDR3, 1600 MHz (via `sudo dmidecode --type memory | less`)

### Software Versions
**Julia**: 1.4.2 (via `julia --version`)
**Python**: 3.6.9 (via `python3 --version`)
**Gurobi**: 9.0.2 (via `echo $GUROBI_HOME`)

Julia and Python packages dependencies are fully specified in `Manifest.toml` and `REQUIREMENTS.txt` respectively.

> :warning: To ensure that you are running Python scripts with the correct dependencies, please execute the scripts with a virtualenv activated, installing the required packages with the virtualenv activated via `pip3 install -r REQUIREMENTS.txt`
> :warning: All `julia` commands should be executed from the `MIPVerify` directory that this README is contained in with the `--project` flag, which ensures that the dependencies specified in `Manifest.toml` are used. If it is necessary to execute the script from a different directory, please use `julia --project=path/to/this/MIPVerify/directory`

### Detailed Instructions
The results for these benchmarks are stored in the [results](./results) directory, organized by benchmark. Each verification script in [scripts/verification](./scripts/verification) saves its results to a `summary.csv` file, which the post-processing scripts in [scripts/reports](./scripts/reports) can convert into a `.txt` file compatible with the report format.

To reproduce the results, remove the folder corresponding to the benchmark that you're trying to reproduce, and follow the following commands.

#### ACASXU-ALL
```
# running verification. output produced at `results/acasxu-all/summary.csv`
julia --project scripts/verification/acasxu-all.jl

# generating `.txt` file compatible with submission format. output produced at `results/acasxu-all.txt`
./scripts/reports/generate_text_submission.py --benchmark_name ACASXU-ALL
```

#### ACASXU-HARD
```
# running verification. output produced at `results/acasxu-hard/summary.csv`
julia --project scripts/verification/acasxu-hard.jl

# generating `.txt` file compatible with submission format. output produced at `results/acasxu-hard.txt`
./scripts/reports/generate_text_submission.py --benchmark_name ACASXU-HARD
```

#### MNIST-OVAL
```
# running verification for each network. output produced in `mnist-oval` directory.
julia --project ./scripts/verification/mnist-net_256x2.jl
julia --project ./scripts/verification/mnist-net_256x4.jl
julia --project ./scripts/verification/mnist-net_256x6.jl

# generating `.txt` file compatible with submission format. output produced at `results/mnist-oval.txt`
./scripts/reports/generate_text_submission.py --benchmark_name MNIST-OVAL

# the `.mat` files used as input in the `networks` directory are produced by running the script
# `./scripts/conversion/extract_onnx_params.py`
```

## Benchmark Details
### ACASXU-ALL
Verifying safety properties first presented in [Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks](https://arxiv.org/abs/1702.01135)

#### Details
  - Timeout: 5 minutes
  - (Network, Property) pairs verified:
    - Properties 1-4: all 45 networks
    - Properties 5-10: the network verified in the original paper.
      - Property 5: Net 1-1
      - Property 6: Net 1-1
      - Property 7: Net 1-9
      - Property 8: Net 2-9
      - Property 9: Net 3-3
      - Property 10: Net 4-5
  - [Data Source](../benchmark/acasxu)

Reference: [[1]](https://github.com/verivital/vnn-comp/issues/2#issuecomment-659422590)

### ACASXU-HARD
A subset of (Network, Property) pairs from [ACASXU-ALL](ACASXU-ALL) identified by [`stanleybak`](https://github.com/pat676) (Stanley Bak) to be challenging, run with a longer timeout.

#### Details
  - Timeout: 6 hours
  - (Network, Property) pairs verified:
    - Property 1: Net 4-6, Net 4-8
    - Property 2: Net 3-3, Net 4-2, Net 4-9, Net 5-3
    - Property 3: Net 3-6, Net 5-1
    - Property 7: Net 1-9
    - Property 9: Net 3-3
  - [Data Source](../benchmark/acasxu)

Reference: [[1]](https://github.com/verivital/vnn-comp/issues/2#issuecomment-626232303)

### MNIST-OVAL
Verifying adversarial robustness for MNIST networks provided by [`pat676`](https://github.com/pat676) (Patrick Henriksen)

#### Details
  - Timeout: 15 minutes
  - Perturbation norms (a.k.a. ε-values): [`[0.02, 0.05]`](https://github.com/verivital/vnn-comp/issues/2#issuecomment-644657954)
  - Images verified: 25 images from the MNIST test set selected by the author.
  - [Data Source](../mnist/oval)

Reference: [[1]](https://github.com/verivital/vnn-comp/issues/2#issuecomment-627487425), [[2]]((https://github.com/verivital/vnn-comp/issues/2#issuecomment-644657954). [[3]](https://github.com/verivital/vnn-comp/issues/2#issuecomment-650565518), [[4]](https://github.com/verivital/vnn-comp/issues/2#issuecomment-656001365)
