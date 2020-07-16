#!/usr/bin/env julia

using Gurobi

include("../../src/nnet.jl")
include("../../src/MIPVerify_patch_acasxu.jl")

## GLOBAL PATHS
NETWORK_PATH = joinpath(@__DIR__, "../../../benchmark/acasxu")

## SOLVERS
# We specify this here so we only acquire the env once
main_timeout = 60 * 5     # seconds, for overall problem
tightening_timeout = 1    # seconds, per non-linearity added

main_solver = GurobiSolver(
    Gurobi.Env(),
    BestObjStop=0,
    BestBdStop=0,
    TimeLimit=main_timeout,
)
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=tightening_timeout)
tightening_algorithm = lp

## INITIALIZE BATCH SOLVE
summary_file_path, summary_dt = initialize_batch_solve(joinpath(@__DIR__, "../../results/acasxu-all"))

## INITIAL, UNTIMED SOLVE
# We select a very simple instance for this and set a very short timeout
verify_property(
    NNet(joinpath(NETWORK_PATH, "ACASXU_run2a_3_1_batch_2000.nnet")),
    "warmup",
    properties[3],
    main_solver,
    tightening_solver,
    tightening_algorithm,
)

## MAIN LOOP
network_property_pairs = Iterators.flatten([
    # Properties 1-4: all networks
    Base.product(1:9, 1:5, 1:4),
    # Properties 5-10: selected networks
    [[1, 1, 5], [1, 1, 6], [9, 1, 7], [9, 2, 8], [3, 3, 9], [5, 4, 10]],
])

for (netid2, netid1, property_id) in network_property_pairs

    network_id = "$(netid1)_$(netid2)"
    property = properties[property_id]

    # Check if a previous solve exists and skip if it does
    if any((summary_dt.NetworkID.==network_id).&(summary_dt.PropertyID.==property_id))
        continue
    end

    name = "ACASXU_run2a_$(network_id)_batch_2000"

    nnet = NNet(joinpath(NETWORK_PATH, "$name.nnet"))

    d = verify_property(
        nnet,
        name,
        property,
        main_solver,
        tightening_solver,
        tightening_algorithm,
    )

    summary_line = generate_csv_summary_line(property_id, network_id, d)

    open(summary_file_path, "a") do file
        writedlm(file, [summary_line], ',')
    end

end
