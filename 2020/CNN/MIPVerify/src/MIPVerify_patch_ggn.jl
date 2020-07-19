using MIPVerify
using Gurobi
using DelimitedFiles
using CSV
using Memento
using DataFrames

function log_correct(nn::NeuralNet, dataset::MIPVerify.LabelledDataset, num_samples::Integer)
    for sample_index in 1:num_samples
        input = MIPVerify.get_image(dataset.images, sample_index)
        actual_label = MIPVerify.get_label(dataset.labels, sample_index)
        predicted_label = (input |> nn |> MIPVerify.get_max_index) - 1
        # img 0 not Correctly Classified , correct_label 3 classified label  5
        # img 1 Correctly Classified  8
        # img 2 Correctly Classified  8
        if actual_label == predicted_label
            println("img $(sample_index-1) Correctly Classified  $actual_label")
        else
            println("img $(sample_index-1) not Correctly Classified , correct_label $actual_label classified label  $predicted_label")
        end
    end
end

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

function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    delimiter::String = "/",
    expected_stride::Integer = 1,
    padding::Padding = SamePadding(),
)::Conv2d

    params = Conv2d(
        param_dict["$layer_name$delimiter$matrix_name"],
        dropdims(param_dict["$layer_name$delimiter$bias_name"], dims = 1),
        expected_stride,
        padding,
    )

    MIPVerify.check_size(params, expected_size)

    return params
end

struct DatasetProps
    n_class::Int
    input_size::Int
    input_channel::Int
    means::AbstractArray{Float64}
    variances::AbstractArray{Float64}
end

function get_dataset_props(
    dataset::String
)::DatasetProps
    if dataset == "MNIST"
        return DatasetProps(10, 28, 1, [0.1307], [0.3081])
    elseif dataset == "CIFAR10"
        return DatasetProps(10, 32, 3, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    else
        throw(ArgumentError("Unknown dataset name."))
    end
end


# TODO (vtjeng): We should be able to extract these values directly from the .onnx file
function get_ConvMed_network(
    param_dict::Dict,
    network_name::String,
    dataset::String,
    width1::Int,
    width2::Int,
    linear_size::Int,
)
    dp = get_dataset_props(dataset)
    conv1 = get_conv_params(param_dict, "blocks.layers.1.conv", (5, 5, dp.input_channel, 16*width1), expected_stride = 2, padding=2, delimiter=".")
    conv2 = get_conv_params(param_dict, "blocks.layers.3.conv", (4, 4, 16*width1, 32*width2), expected_stride = 2, padding=1, delimiter=".")
    linear1 = get_matrix_params(param_dict, "blocks.layers.6.linear", (32*width2*floor(Int, dp.input_size / 4)^2, linear_size), delimiter=".")
    linear2 = get_matrix_params(param_dict, "blocks.layers.8.linear", (linear_size, dp.n_class), delimiter=".")

    nnparams = Sequential([
        # https://github.com/eth-sri/colt/blob/20f30b073558ae80e5e726515998c1f31d48b6c6/code/loaders.py#L11-L13
        Normalize(dp.means, dp.variances),
        conv1,
        ReLU(interval_arithmetic),
        conv2,
        ReLU(),
        Flatten([1, 3, 2, 4]),
        linear1,
        ReLU(),
        linear2], "$(network_name)"
    )
    return nnparams
end

# TODO (vtjeng): We should be able to extract these values directly from the .onnx file
function get_ConvMedBig_network(
    param_dict::Dict,
    network_name::String,
    dataset::String,
    width1::Int,
    width2::Int,
    width3::Int,
    linear_size::Int,
)
    dp = get_dataset_props(dataset)

    conv1 = get_conv_params(param_dict, "blocks.layers.1.conv", (3, 3, dp.input_channel, 16*width1), expected_stride = 1, padding=1, delimiter=".")
    conv2 = get_conv_params(param_dict, "blocks.layers.3.conv", (4, 4, 16*width1, 16*width2), expected_stride = 2, padding=1, delimiter=".")
    conv3 = get_conv_params(param_dict, "blocks.layers.5.conv", (4, 4, 16*width2, 32*width3), expected_stride = 2, padding=1, delimiter=".")
    linear1 = get_matrix_params(param_dict, "blocks.layers.8.linear", (32*width3*floor(Int, dp.input_size / 4)^2, linear_size), delimiter=".")
    linear2 = get_matrix_params(param_dict, "blocks.layers.10.linear", (linear_size, dp.n_class), delimiter=".")

    nnparams = Sequential([
        # https://github.com/eth-sri/colt/blob/20f30b073558ae80e5e726515998c1f31d48b6c6/code/loaders.py#L11-L13
        Normalize(dp.means, dp.variances),
        conv1,
        ReLU(interval_arithmetic),
        conv2,
        ReLU(),
        conv3,
        ReLU(),
        Flatten([1, 3, 2, 4]),
        linear1,
        ReLU(),
        linear2], "$(network_name)"
    )
    return nnparams
end

## main helper that we call
function verify(
    nnparams::MIPVerify.NeuralNet,
    eps::Float64,
    target_indices::AbstractArray{<:Integer},
    dataset_name::String,
    timeout_secs::Int,
    save_path::String,
    )
    dataset = read_datasets(dataset_name)

    println("Fraction correct of first 100 is $(frac_correct(nnparams, dataset.test, 100))")

    MIPVerify.setloglevel!("info")

    println("Carrying out an initial, untimed solve using a tiny perturbation radius. This was put in place because we observed that the first solve was disproportionately slow and not representative of actual solve performance.")
    d = find_adversarial_example(
        nnparams,
        MIPVerify.get_image(dataset.test.images, 1),
        1,
        GurobiSolver(TimeLimit=1),
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.0001),
        rebuild = true,
        tightening_algorithm = interval_arithmetic,
        cache_model = false,
        adversarial_example_objective = MIPVerify.worst,
    )

    println("Carrying out actual timed solves.")
    batch_find_untargeted_attack(
        nnparams,
        dataset.test,
        target_indices,
        GurobiSolver(Gurobi.Env(), BestObjStop=0, BestBdStop=0, TimeLimit=timeout_secs),
        pp = MIPVerify.LInfNormBoundedPerturbationFamily(eps),
        norm_order=Inf,
        rebuild=true,
        solve_rerun_option = MIPVerify.never,
        tightening_algorithm=lp,
        tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=5),
        cache_model = false,
        solve_if_predicted_in_targeted = false,
        save_path = save_path,
        adversarial_example_objective = MIPVerify.worst,
    )
end

## monkey patch functions
function initialize_batch_solve(
    save_path::String,
    nn::NeuralNet,
    pp::MIPVerify.PerturbationFamily,
)

    summary_file_name = "summary.csv"

    main_path = joinpath(save_path, "$(nn.UUID)")

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
    if !(d[:PredictedIndex] in d[:TargetIndexes]) || solve_if_predicted_in_targeted
        r = MIPVerify.extract_results_for_save(d)
        summary_line = generate_csv_summary_line(sample_number, r)
    else
        summary_line = generate_csv_summary_line_optimal(sample_number, d)
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
            throw(ArgumentError("Unexpected pair of objective value $val and bound $bd."))
        end
    elseif s == :UserLimit
        return :Timeout
    elseif s == :Error
        return :Error
    elseif s == :Infeasible || s == :InfeasibleOrUnbounded
        # This happens if we statically determine that the target_selection can never be the maximum
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
        :Misclassified,
        0,
        0,
        0,
        r[:TotalTime],
    ] .|> string
end

function batch_find_untargeted_attack(
    nn::NeuralNet,
    dataset::MIPVerify.LabelledDataset,
    target_indices::AbstractArray{<:Integer},
    main_solver;
    save_path::String = ".",
    solve_rerun_option::MIPVerify.SolveRerunOption = MIPVerify.never,
    pp::MIPVerify.PerturbationFamily = MIPVerify.UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tolerance::Real = 0.0,
    rebuild = false,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = MIPVerify.DEFAULT_TIGHTENING_ALGORITHM,
    tightening_solver = MIPVerify.get_default_tightening_solver(
        main_solver,
    ),
    cache_model = true,
    solve_if_predicted_in_targeted = true,
    adversarial_example_objective = MIPVerify.closest,
)::Nothing

    MIPVerify.verify_target_indices(target_indices, dataset)
    (summary_file_path, dt) =
        initialize_batch_solve(save_path, nn, pp)

    for sample_number in target_indices
        should_run = MIPVerify.run_on_sample_for_untargeted_attack(sample_number, dt, solve_rerun_option)
        if should_run
            # TODO (vtjeng): change function signature for get_image and get_label
            Memento.info(MIPVerify.LOGGER, "Working on index $(sample_number)")
            input = MIPVerify.get_image(dataset.images, sample_number)
            true_one_indexed_label = MIPVerify.get_label(dataset.labels, sample_number) + 1
            d = find_adversarial_example(
                nn,
                input,
                true_one_indexed_label,
                main_solver,
                invert_target_selection = true,
                pp = pp,
                norm_order = norm_order,
                tolerance = tolerance,
                rebuild = rebuild,
                tightening_algorithm = tightening_algorithm,
                tightening_solver = tightening_solver,
                cache_model = cache_model,
                solve_if_predicted_in_targeted = solve_if_predicted_in_targeted,
                adversarial_example_objective = adversarial_example_objective,
            )

            save_to_disk(
                sample_number,
                summary_file_path,
                d,
                solve_if_predicted_in_targeted,
            )
        end
    end
    return nothing
end
