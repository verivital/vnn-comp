#!/usr/bin/env julia
using MAT
include("../../src/MIPVerify_patch_ggn.jl")

param_dict = matread(joinpath(@__DIR__, "../../networks/mnist_0.3.mat"))

nnparams = get_ConvMed_network(
    param_dict,
    "mnist_0.3",
    "MNIST",
    2, 4, 250
)

verify(
    nnparams,
    0.3,
    1:100,
    "MNIST",
    5*60,
    joinpath(@__DIR__, "../../results/ggn-cnn"),
)
