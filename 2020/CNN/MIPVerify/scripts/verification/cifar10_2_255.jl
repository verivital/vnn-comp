#!/usr/bin/env julia
using MAT
include("../../src/MIPVerify_patch_ggn.jl")


# We reported a timeout for each of the benchmarks here as the verifier encountered time-outs
# for most of the pre-solves anyway.

# param_dict = matread(joinpath(@__DIR__, "../../networks/cifar10_2_255.mat"))
#
# nnparams = get_ConvMedBig_network(
#     param_dict,
#     "cifar10_2_255",
#     "CIFAR10",
#     2, 2, 4, 250
# )
#
# verify(
#     nnparams,
#     2/255,
#     1:100,
#     "CIFAR10",
#     5*60,
#     joinpath(@__DIR__, "../../results/ggn-cnn"),
# )
