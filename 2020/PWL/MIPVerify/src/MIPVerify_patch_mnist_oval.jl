using MIPVerify
using DelimitedFiles
using CSV
using DataFrames

MNIST_OVAL_RESOURCE_DIR = "../../benchmark/mnist/oval"
RESULT_OUTPUT_DIR = "../results/mnist-oval"
TIMEOUT = 15*60

function get_matrix_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{2,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    delimiter::String = "/",
)::Linear

    params = Linear(
        param_dict["$layer_name$delimiter$matrix_name"],
        dropdims(param_dict["$layer_name$delimiter$bias_name"], dims = 1),
    )

    MIPVerify.check_size(params, expected_size)

    return params
end

function initialize_batch_solve(
    save_path::String,
    nn::NeuralNet,
    pp::MIPVerify.PerturbationFamily,
)

    summary_file_name = "summary.csv"

    main_path = joinpath(save_path, "$(nn.UUID)_$(pp)")

    main_path |> MIPVerify.mkpath_if_not_present

    summary_file_path = joinpath(main_path, summary_file_name)
    summary_file_path |> create_summary_file_if_not_present

    dt = DataFrame!(CSV.File(summary_file_path))
    return (summary_file_path, dt)
end

function save_to_disk(
    sample_number::Integer,
    summary_file_path::String,
    d::Dict,
    solve_if_predicted_in_targeted::Bool,
)
    r = MIPVerify.extract_results_for_save(d)
    if !(r[:PredictedIndex] in r[:TargetIndexes]) || solve_if_predicted_in_targeted
        summary_line = generate_csv_summary_line(sample_number, r)
    else
        summary_line = generate_csv_summary_line_optimal(sample_number, r)
    end

    open(summary_file_path, "a") do file
        writedlm(file, [summary_line], ',')
    end
end

function create_summary_file_if_not_present(summary_file_path::String)
    if !isfile(summary_file_path)
        summary_header_line = [
            "SampleNumber",
            "VerificationResult",
            "ObjectiveValue",
            "ObjectiveBound",
            "MainSolveTime",
            "TotalTime",
        ]

        open(summary_file_path, "w") do file
            writedlm(file, [summary_header_line], ',')
        end
    end
end

function process_solve_status(r::Dict)
    s = r[:SolveStatus]
    if s == :UserObjLimit || s == :Optimal
        val = r[:ObjectiveValue]
        bd = r[:ObjectiveBound]
        if val > 0
            return :SAT
        elseif bd <= 0
            return :UNSAT
        else
            throw(ArgumentError("Unexpected pair of objective value $val and bound $bd."))MNIST_OVAL_RESOURCE_DIR
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
    sample_number::Integer,
    r::Dict,
)
    [
        sample_number,
        process_solve_status(r),
        r[:ObjectiveValue],
        r[:ObjectiveBound],
        r[:SolveTime],
        r[:TotalTime],
    ] .|> string
end

function generate_csv_summary_line_optimal(
    sample_number::Integer,
    r::Dict,
)
    [
        sample_number,
        :SAT,
        0,
        0,
        0,
        r[:TotalTime],
    ] .|> string
end

function read_csv_line(path, type)
    s = open(path) do file
        read(file, String)
    end

    readdlm(
        IOBuffer(chop(s, tail=1)),  # remove trailing comma
        ',',
        type
    )
end

# https://julialang.org/blog/2018/07/iterators-in-julia-0.7/
Base.@kwdef struct pat676MNISTGenerator
    image_folder = joinpath(@__DIR__, "$(MNIST_OVAL_RESOURCE_DIR)/mnist_images")[:]
    labels_file = joinpath(@__DIR__, "$(MNIST_OVAL_RESOURCE_DIR)/mnist_images/labels")
    start::Int = 1
    length::Int = 25
end

function Base.iterate(iter::pat676MNISTGenerator, state=0)
    count = state

    if count >= iter.length
        return nothing
    end

    sample_number = state + 1
    image = read_csv_line("$(iter.image_folder)/image$sample_number", Int)[:]/255
    # image = transpose(reshape(read_csv_line("$(iter.image_folder)/image$sample_number", Int), (28, 28)))[:]/255
    label = read_csv_line(iter.labels_file, Int)[sample_number] + 1

    return ((image, label), count+1)
end

Base.length(iter::pat676MNISTGenerator) = iter.length

function frac_correct(nn, generator)
    num_correct = 0
    for (image, true_one_indexed_label) in generator
        predicted_label = image |> nn |> MIPVerify.get_max_index
        if predicted_label == true_one_indexed_label
            num_correct += 1
        end
    end
    return num_correct / length(generator)
end

function main_solve_helper(nn, ta, eps, generator)
    main_solver = GurobiSolver(Gurobi.Env(), BestObjStop=0, BestBdStop=0, TimeLimit=TIMEOUT)
    pp = MIPVerify.LInfNormBoundedPerturbationFamily(eps)
    norm_order=Inf
    rebuild=true
    tightening_algorithm=ta
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=5)
    cache_model = false
    solve_if_predicted_in_targeted = false

    (summary_file_path, summary_dt) = initialize_batch_solve(joinpath(@__DIR__, RESULT_OUTPUT_DIR), nn, pp)

    image, true_one_indexed_label = first(generator())
    # Initial untimed solve
    println("Carrying out an initial, untimed solve using a tiny perturbation radius. This was put in place because we observed that the first solve was disproportionately slow and not representative of actual solve performance.")
    find_adversarial_example(
        nn,
        image,
        true_one_indexed_label,
        main_solver,
        invert_target_selection = true,
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(eps/1000),
        norm_order = norm_order,
        rebuild = rebuild,
        adversarial_example_objective = MIPVerify.worst,
        tightening_algorithm = tightening_algorithm,
        tightening_solver = tightening_solver,
        cache_model = cache_model,
        solve_if_predicted_in_targeted = solve_if_predicted_in_targeted,
    )

    println("Carrying out actual timed solves.")
    # actual timed solves
    for (sample_number, (image, true_one_indexed_label)) in enumerate(generator())
        if (sample_number in summary_dt[!, :SampleNumber])
            continue
        end

        d = find_adversarial_example(
            nn,
            image,
            true_one_indexed_label,
            main_solver,
            invert_target_selection = true,
            pp = pp,
            norm_order = norm_order,
            rebuild = rebuild,
            adversarial_example_objective = MIPVerify.worst,
            tightening_algorithm = tightening_algorithm,
            tightening_solver = tightening_solver,
            cache_model = cache_model,
            solve_if_predicted_in_targeted = solve_if_predicted_in_targeted,
        )

        save_to_disk(
            sample_number,
            summary_file_path,
            d,
            solve_if_predicted_in_targeted,
        )
    end
end
