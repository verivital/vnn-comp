#!/usr/bin/env julia
using MAT
include("../../src/MIPVerify_patch_ggn.jl")

param_dict = matread(joinpath(@__DIR__, "../../networks/mnist_0.1.mat"))

nnparams = get_ConvMed_network(
    param_dict,
    "mnist_0.1",
    "MNIST",
    2, 2, 100
)

verify(
    nnparams,
    0.1,
    1:100,
    "MNIST",
    1*60,
    joinpath(@__DIR__, "../../results/ggn-cnn"),
)
