#!/usr/bin/env julia

using MIPVerify
using MAT
using Gurobi
include("../../src/MIPVerify_patch_mnist_oval.jl")

param_dict = matread(joinpath(@__DIR__, "../../networks/mnist-net_256x6.mat"))

linear1 = get_matrix_params(param_dict, "layers.0", (784, 256), delimiter=".")
linear2 = get_matrix_params(param_dict, "layers.2", (256, 256), delimiter=".")
linear3 = get_matrix_params(param_dict, "layers.4", (256, 256), delimiter=".")
linear4 = get_matrix_params(param_dict, "layers.6", (256, 256), delimiter=".")
linear5 = get_matrix_params(param_dict, "layers.8", (256, 256), delimiter=".")
linear6 = get_matrix_params(param_dict, "layers.10", (256, 256), delimiter=".")
linear7 = get_matrix_params(param_dict, "layers.12", (256, 10), delimiter=".")

nn = Sequential([
    linear1,
    ReLU(interval_arithmetic),
    linear2,
    ReLU(),
    linear3,
    ReLU(),
    linear4,
    ReLU(),
    linear5,
    ReLU(),
    linear6,
    ReLU(),
    linear7,
], "mnist-net_256x6")

MIPVerify.setloglevel!("info")

println("Fraction correct of 25 samples: $(frac_correct(nn, pat676MNISTGenerator()))")

main_solve_helper(nn, lp, 0.02, pat676MNISTGenerator)
main_solve_helper(nn, lp, 0.05, pat676MNISTGenerator)
