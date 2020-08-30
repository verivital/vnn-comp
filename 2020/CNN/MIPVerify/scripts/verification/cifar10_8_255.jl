#!/usr/bin/env julia
using MAT
include("../../src/MIPVerify_patch_ggn.jl")

param_dict = matread(joinpath(@__DIR__, "../../networks/cifar10_8_255.mat"))

nnparams = get_ConvMed_network(param_dict, "cifar10_8_255", "CIFAR10", 2, 4, 250)

verify(
    nnparams,
    8 / 255,
    1:100,
    "CIFAR10",
    5 * 60,
    joinpath(@__DIR__, "../../results/ggn-cnn"),
)
