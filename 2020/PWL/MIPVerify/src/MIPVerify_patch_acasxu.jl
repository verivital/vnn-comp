using Memento
using CSV
using DataFrames

MIPVerify.setloglevel!("info")

function create_summary_file_if_not_present(summary_file_path::String)
    if !isfile(summary_file_path)
        summary_header_line = [
            "PropertyID",
            "NetworkID",
            "VerificationResult",
            "InternalSolveStatus",
            "ObjectiveValue",
            "ObjectiveBound",
            "InputValue",
            "OutputValue",
            "MainSolveTime",
            "TotalTime",
        ]

        open(summary_file_path, "w") do file
            writedlm(file, [summary_header_line], ',')
        end
    end
end

function process_verification_status(r::Dict)
    s = r[:SolveStatus]
    if s == :UserObjLimit || s == :Optimal
        val = r[:ObjectiveValue]
        bd = r[:ObjectiveBound]
        if bd < val
            @warn "Unexpected pair of objective value $val and bound $bd; reported bound is lower than reported value for a maximization problem."
            return :BoundValueMismatchError
        end
        if val > 0
            return :SAT
        elseif bd <= 0
            return :UNSAT
        else
            @warn "Unexpected pair of objective value $val and bound $bd; optimization should not have stopped here."
            return :BoundValueMismatchError
        end
    elseif s == :UserLimit
        return :Timeout
    elseif s == :Error
        return :Error
    elseif s == :Infeasible || s == :InfeasibleOrUnbounded
        return :UNSAT
    else
        throw(ArgumentError("Unknown solve status $s"))
    end
end

function generate_csv_summary_line(
    property_id::Integer,
    network_id::String,
    r::Dict,
)
    verification_status = process_verification_status(r)

    input_value = NaN
    output_value = NaN
    if verification_status == :SAT
        input_value = r[:InputValue]
        output_value = r[:OutputValue]
    end

    [
        property_id,
        network_id,
        verification_status,
        r[:SolveStatus],
        r[:ObjectiveValue],
        r[:ObjectiveBound],
        input_value,
        output_value,
        r[:SolveTime],
        r[:TotalTime],
    ] .|> string
end

function initialize_batch_solve(save_path::String)
    summary_file_name = "summary.csv"

    save_path |> MIPVerify.mkpath_if_not_present

    summary_file_path = joinpath(save_path, summary_file_name)
    summary_file_path |> create_summary_file_if_not_present

    summary_dt = DataFrame!(CSV.File(summary_file_path))
    return (summary_file_path, summary_dt)
end

function verify_property(
    nnet::NNet,
    name::String,
    property::VerificationProperty,
    main_solver,
    tightening_solver,
    tightening_algorithm::MIPVerify.TighteningAlgorithm,
)
    nn = get_nn_params(nnet, name)

    total_time = @elapsed begin
        # dictionary to store results
        d = Dict{Symbol, Any}()

        m = Model()
        d[:Model] = m

        # use the solver that we want to use for the bounds tightening
        JuMP.setsolver(m, tightening_solver)

        # set the tightening algorithm to that specified
        m.ext[:MIPVerify] = MIPVerify.MIPVerifyExt(tightening_algorithm)

        # v_in is the variable representing the actual range of input values
        v_in = @variable(
            m,
            x[i=1:nnet.input_size]
        )

        # these input constraints need to be set before we feed the bounds
        # forward through the network via the call nn(v_in)
        property.set_input_constraints(v_in, nnet)

        v_out = nn(v_in)

        # Introduce an additional variable since Gurobi ignores constant terms in objective, but we explicitly need these to stop early: https://github.com/jump-dev/Gurobi.jl/issues/111
        v_obj_expr = property.get_objective(v_out, nnet)
        v_obj = @variable(m)
        @constraint(m, v_obj == v_obj_expr)

        # Provide input warm start
        fill!(m.colVal, NaN)
        warm_start = property.get_warm_start(v_in, nnet)
        Memento.info(
            MIPVerify.LOGGER,
            "Warm start values: $(denormalize_input(warm_start, nnet)).",
        )
        for (v, v_val) in zip(v_in, warm_start)
            setvalue(v, v_val)
        end

        # Use the main solver
        JuMP.setsolver(m, main_solver)

        @objective(m, Max, v_obj)

        d[:SolveStatus] = solve(m)
    end

    d[:SolveTime] = getsolvetime(m)
    d[:TotalTime] = total_time
    d[:ObjectiveBound] = getobjbound(m)
    d[:ObjectiveValue] = getobjectivevalue(m)
    d[:InputValue] = denormalize_input(map(getvalue, v_in), nnet)
    d[:OutputValue] = denormalize_output(map(getvalue, v_out), nnet)

    return d
end
