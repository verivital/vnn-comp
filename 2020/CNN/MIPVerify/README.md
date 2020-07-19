# MIPVerify
This folder contains code to reproduce the results for the [MIPVerify](https://github.com/vtjeng/MIPVerify.jl) tool on the [benchmark for convolutional neural networks](https://github.com/verivital/vnn-comp/issues/3).

See the corresponding See corresponding [README.md](../../PWL/MIPVerify/README.md) file for the PWL set of benchmarks for manual installation, and system specifications and software versions for reproducing results.

## Reproducing Results
### Detailed Instructions
The results for these benchmarks are stored in the [results](./results) directory, organized by benchmark. Each verification script in [scripts/verification](./scripts/verification) saves its results to a `summary.csv` file, which the post-processing scripts in [scripts/reports](./scripts/reports) can convert into a `.txt` file compatible with the report format.

To reproduce the results, remove the folder corresponding to the benchmark that you're trying to reproduce, and follow the following commands.

#### GGN-CNN
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
