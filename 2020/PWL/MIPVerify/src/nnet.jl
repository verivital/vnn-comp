using MIPVerify
using DelimitedFiles
using JuMP
using ConditionalJuMP

"""
Conversion from .nnet format to network format used by MIPVerify
Properties specified in Reluplex paper in a format usable by the verification script
"""

"""
https://github.com/sisl/NNet/blob/master/julia/nnet.jl#L1 improved with clear error checking

Custom type that represents a fully connected ReLU network from a .nnet file
Args:
    file (string): A .nnet file to load
Attributes:
    num_layers (int): Number of weight matrices or bias vectors in neural network
    layer_sizes (list of ints): Size of input layer, hidden layers, and output layer
    input_size (int): Size of input
    output_size (int): Size of output
    mins (list of floats): Minimum values of inputs
    maxes (list of floats): Maximum values of inputs
    means (list of floats): Means of inputs and mean of outputs
    ranges (list of floats): Ranges of inputs and range of outputs
    weights (list of arrays): Weight matrices in network
    biases (list of arrays): Bias vectors in network
"""
struct NNet
    weights::Array{Array{Float64, 2},1}
    biases::Array{Array{Float64, 1},1}
    num_layers::Int32
    layer_sizes::Array{Int32,1}
    input_size::Int32
    output_size::Int32
    mins::Array{Float64,1}
    maxes::Array{Float64,1}
    means::Array{Float64,1}
    ranges::Array{Float64,1}
end

function read_nnet_line(f::IOStream, T::Type)
    line = readline(f)
    return parse_nnet_line(line, T)
end

function parse_nnet_line(line::String, T::Type)
    # Verify that we have a trailing comma, and drop it.
    @assert line[end] == ','
    return readdlm(IOBuffer(line[1:end-1]), ',', T)
end

function NNet(filename::AbstractString)
    open(filename) do f
        # Skip any header lines
        line = readline(f)
        while line[1:2]=="//"
            line = readline(f)
        end

        # Read information about the neural network
        # We have to put the line back in an IOBuffer
        num_layers, input_size, output_size = parse_nnet_line(line, Int32)

        layer_sizes = read_nnet_line(f, Int32)[:]
        @assert size(layer_sizes) == (num_layers + 1, )
        @assert layer_sizes[1] == input_size
        @assert layer_sizes[end] == output_size

        # Skip unused line
        readline(f)

        mins = read_nnet_line(f, Float64)[:]
        @assert size(mins) == (input_size, )

        maxes = read_nnet_line(f, Float64)[:]
        @assert size(maxes) == (input_size, )

        means = read_nnet_line(f, Float64)[:]
        @assert size(means) == (input_size + 1, )

        ranges = read_nnet_line(f, Float64)[:]
        @assert size(ranges) == (input_size + 1, )

        weights = Matrix{Float64}[]
        biases = Vector{Float64}[]
        for layer_idx in 1:num_layers
            layer_input_size = layer_sizes[layer_idx]
            layer_output_size = layer_sizes[layer_idx + 1]
            weight = vcat([read_nnet_line(f, Float64) for _ in 1:layer_output_size]...)
            @assert size(weight) == (layer_output_size, layer_input_size)
            raw_bias = hcat([read_nnet_line(f, Float64) for _ in 1:layer_output_size]...)
            bias = raw_bias[:]
            @assert size(bias) == (layer_output_size,)
            push!(weights, weight)
            push!(biases, bias)
        end

        params = [weights, biases, num_layers, layer_sizes, input_size, output_size, mins, maxes, means, ranges]

        # Validate that we have read all of the data
        @assert eof(f)
        return NNet(params...)
    end
end

function get_nn_params(nnet::NNet, name::String)::MIPVerify.NeuralNet
    """
    Verified via example provided in https://github.com/NeuralNetworkVerification/Marabou/issues/225
    """
    @assert length(nnet.weights) == length(nnet.biases)

    layers = MIPVerify.Layer[]
    # Normalize input
    # https://github.com/sisl/NNet/blob/4411dd47621489f44062ca96898b8cebd722c7c8/julia/nnet.jl#L148-L158
    # push!(layers, Normalize(nnet.means[1:nnet.input_size], nnet.ranges[1:nnet.input_size]))

    # Add fully connected layers
    for (layer_idx, (weight, bias)) in enumerate(zip(nnet.weights, nnet.biases))
        push!(layers, Linear(Array(transpose(weight)), bias))
        if layer_idx != nnet.num_layers
            push!(layers, layer_idx == 1 ? ReLU(interval_arithmetic) : ReLU())
        end
    end

    # Undo output normalization
    # https://github.com/sisl/NNet/blob/4411dd47621489f44062ca96898b8cebd722c7c8/julia/nnet.jl#L167-L171
    # ax + b = (x - (-b/a))/(1/a)
    # μ = -nnet.means[end]/nnet.ranges[end]
    # σ = 1/nnet.ranges[end]
    # push!(layers, Normalize(μ*ones(nnet.output_size), σ*ones(nnet.output_size)))
    return Sequential(layers, name)
end

struct VerificationProperty
    set_input_constraints::Function
    """
    Objective to be maximized.

    The objective must satisfy the following conditions:
      - An assignment of variables with positive objective value corresponds to a violation of the
        property ("unsafe" / "SAT").
      - Alternatively, if no such assignment exists (i.e. the objective is provably negative), the
        property is never violated ("safe" / "UNSAT")
    """
    get_objective::Function
    get_warm_start::Function
end

function normalize_input(x::Real, nnet::NNet, index::Int)::Float64
    @assert index >= 1
    @assert index <= nnet.input_size
    return (x - nnet.means[index])/nnet.ranges[index]
end

function normalize_output(x::Real, nnet::NNet)::Float64
    return (x - nnet.means[end])/nnet.ranges[end]
end

function denormalize_output(xs_normalized::Array{T, 1}, nnet::NNet)::Array{Float64, 1} where T<:Real
    return xs_normalized*nnet.ranges[end] .+ nnet.means[end]
end

function normalize_input(xs::Array{T, 1}, nnet::NNet)::Array{Float64, 1} where T<:Real
    @assert length(xs) == nnet.input_size
    return (xs - nnet.means[1:nnet.input_size])./nnet.ranges[1:nnet.input_size]
end

function denormalize_input(xs_normalized::Array{T, 1}, nnet::NNet)::Array{Float64, 1} where T<:Real
    @assert length(xs_normalized) == nnet.input_size
    return xs_normalized.*nnet.ranges[1:nnet.input_size] + nnet.means[1:nnet.input_size]
end

"""
input: [ρ, θ, ψ, v_own, v_int]
output: [COC, weak left, weak right, strong left, strong right]
mins: [0.0, -3.141593, -3.141593, 100.0, 0.0]
maxes: [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]
"""

function set_standard_input_constraints(v_input::Array{JuMP.Variable, 1}, lower_bounds::Array{Float64, 1}, upper_bounds::Array{Float64, 1}, nnet::NNet)
    @assert length(v_input) == length(lower_bounds)
    @assert length(v_input) == length(upper_bounds)
    @assert all(lower_bounds .<= upper_bounds) "Lower bounds $lower_bounds must be elementwise no greater than upper bounds $upper_bounds"

    setupperbound.(v_input, normalize_input(min.(upper_bounds, nnet.maxes), nnet))
    setlowerbound.(v_input, normalize_input(max.(lower_bounds, nnet.mins), nnet))
end

function get_minimality_objective(v_output::Array{T, 1}, target_indices::AbstractArray{Int, 1}) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the minimum value of the target_indices *is* smaller than the minimum
    value of the non-target_indices.

    Returns an objective to be Maximized that is positive if and only if the output property
    *is not* True.
    """
    min_off_target = -MIPVerify.maximum(-v_output[setdiff(1:end, target_indices)])
    # since we are maximizing the objective, `min_target` only needs to be restricted to be no
    # larger than the minimum.
    min_target = -MIPVerify.maximum_ge(-v_output[target_indices])

    return min_target - min_off_target
end

function get_default_warm_start(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    return (lowerbound.(v_input) + upperbound.(v_input))./2
end

## PROPERTY 1
function set_p1_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    ρ ≥ 55947.691
    v_own ≥ 1145
    v_int ≤ 60
    """
    set_standard_input_constraints(
        v_input,
        [55947.691, -Inf, -Inf, 1145, -Inf],
        [Inf, Inf, Inf, Inf, 60],
        nnet,
    )
end

function get_p1_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the score for COC is at most 1500.

    This is positive when the score for COC is above 1500.
    """
    return (v_output[1] - normalize_output(1500, nnet))
end

property1 = VerificationProperty(
    set_p1_input_constraints,
    get_p1_objective,
    get_default_warm_start,
)



## PROPERTY 2
function get_p2_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the score for COC *is not* the maximal score

    Objective: The difference between a) the score for COC b) the maximum of all other scores.
    This is positive when the score for COC *is* the maximal score.
    """
    return v_output[1] - MIPVerify.maximum_ge(v_output[2:end])
end

property2 = VerificationProperty(
    # same input constraints as property 1
    set_p1_input_constraints,
    get_p2_objective,
    get_default_warm_start,
)



## PROPERTY 3
function set_p3_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    1500 ≤ ρ ≤ 1800
    −0.06 ≤ θ ≤ 0.06
    ψ ≥ 3.10
    vown ≥ 980
    vint ≥ 960
    """
    set_standard_input_constraints(
        v_input,
        [1500, -0.06, 3.10, 980, 960],
        [1800, 0.06, Inf, Inf, Inf],
        nnet
    )
end

function get_p3_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the score for COC *is not* the minimal score.

    (i.e. the score for one of the rest of the indices *is* minimal)
    """
    return get_minimality_objective(v_output, 2:5)
end

property3 = VerificationProperty(
    set_p3_input_constraints,
    get_p3_objective,
    get_default_warm_start,
)



## PROPERTY 4
function set_p4_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    1500 ≤ ρ ≤ 1800
    −0.06 ≤ θ ≤ 0.06
    ψ = 0
    vown ≥ 1000
    700 ≤ vint ≤ 800
    """
    set_standard_input_constraints(
        v_input,
        [1500, -0.06, 0, 1000, 700],
        [1800, 0.06, 0, Inf, 800],
        nnet,
    )
end

property4 = VerificationProperty(
    # same objective as for p3
    set_p4_input_constraints,
    get_p3_objective,
    get_default_warm_start,
)



## PROPERTY 5
function set_p5_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    250 ≤ ρ ≤ 400
    0.2 ≤ θ ≤ 0.4
    −3.141592 ≤ ψ ≤ −3.141592 + 0.005
    100 ≤ vown ≤ 400
    0 ≤ vint ≤ 400
    """
    set_standard_input_constraints(
        v_input,
        [250, 0.2, -3.141592, 100, 0],
        [400, 0.4, -3.141592 + 0.005, 400, 400],
        nnet
    )
end

function get_p5_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the score for "strong right" *is* the minimal score.
    """
    return get_minimality_objective(v_output, [5])
end

property5 = VerificationProperty(
    set_p5_input_constraints,
    get_p5_objective,
    get_default_warm_start,
)



## PROPERTY 6
function set_p6_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    12000 ≤ ρ ≤ 62000
    (0.7 ≤ θ ≤ 3.141592) ∨ (−3.141592 ≤ θ ≤ −0.7)
    −3.141592 ≤ ψ ≤ −3.141592 + 0.005
    100 ≤ vown ≤ 1200
    0 ≤ vint ≤ 1200
    """
    set_standard_input_constraints(
        v_input,
        [12000, -3.141592, -3.141592, 100, 0],
        [62000, 3.141592, -3.141592 + 0.005, 1200, 1200],
        nnet
    )

    θ = v_input[2]
    m = ConditionalJuMP.getmodel(θ)
    # abs(x) = -x+2*max(x, 0)
    # Also, we take advantage of the fact that the normalization here is symmetric
    @constraint(m, -θ+2*MIPVerify.relu(θ) >= normalize_input(0.7, nnet, 2))
end

function get_p6_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the score for "COC" *is* the minimal score.
    """
    return get_minimality_objective(v_output, [1])
end

function get_p6_warm_start(v_input::Array{JuMP.Variable, 1}, nnet::NNet)::Array{Float64, 1}
    warm_start = get_default_warm_start(v_input, nnet)
    # can't just use midpoint of upper and lower bounds as we have a disjunction here
    warm_start[2] = normalize_input(1.0, nnet, 2)
    return warm_start
end

property6 = VerificationProperty(
    set_p6_input_constraints,
    get_p6_objective,
    get_p6_warm_start,
)



## PROPERTY 7
function set_p7_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    We are meant to search the whole input space here.
    """
    set_standard_input_constraints(
        v_input,
        -Inf*ones(5),
        Inf*ones(5),
        nnet,
    )
end

function get_p7_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the scores for "strong right" and "strong left" *are never* the minimal scores.

    (i.e. one of the remainder are minimal)
    """
    return get_minimality_objective(v_output, 1:3)
end

property7 = VerificationProperty(
    set_p7_input_constraints,
    get_p7_objective,
    # we've chosen a combination of initial values that is likely to trigger a strong turn.
    (v_input, nnet) -> normalize_input([325.0, 0.3, -3.139092, 250.0, 200.0], nnet),
)



## PROPERTY 8
function set_p8_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    0 ≤ ρ ≤ 60760
    -3.141592 ≤ θ ≤ -0.75·3.141592
    −0.1 ≤ ψ ≤ 0.1
    600 ≤ vown ≤ 1200
    600 ≤ vint ≤ 1200
    """
    set_standard_input_constraints(
        v_input,
        [0, -3.141592, -0.1, 600, 600],
        [60760, -0.75*3.141592, 0.1, 1200, 1200],
        nnet,
    )
end

function get_p8_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the scores for "weak left" or the score for COC is minimal.
    """
    return get_minimality_objective(v_output, 1:2)
end

property8 = VerificationProperty(
    set_p8_input_constraints,
    get_p8_objective,
    get_default_warm_start,
)



## PROPERTY 9
function set_p9_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    2000 ≤ ρ ≤ 7000
    −0.4 ≤ θ ≤ −0.14
    −3.141592 ≤ ψ ≤ −3.141592 + 0.01
    100 ≤ vown ≤ 150
    0 ≤ vint ≤ 150
    """
    set_standard_input_constraints(
        v_input,
        [2000, -0.4, -3.141592, 100, 0],
        [7000, -0.14, -3.141592 + 0.01, 150, 150],
        nnet,
    )
end

function get_p9_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the score for “strong left” is minimal
    """
    return get_minimality_objective(v_output, [4])
end

property9 = VerificationProperty(
    set_p9_input_constraints,
    get_p9_objective,
    get_default_warm_start,
)



## PROPERTY 10
function set_p10_input_constraints(v_input::Array{JuMP.Variable, 1}, nnet::NNet)
    """
    36000 ≤ ρ ≤ 60760
    0.7 ≤ θ ≤ 3.141592
    −3.141592 ≤ ψ ≤ −3.141592 + 0.01
    900 ≤ vown ≤ 1200
    600 ≤ vint ≤ 1200
    """
    set_standard_input_constraints(
        v_input,
        [36000, 0.7, -3.141592, 900, 600],
        [60760, 3.141592, -3.141592 + 0.01, 1200, 1200],
        nnet,
    )
end

function get_p10_objective(v_output::Array{T, 1}, nnet::NNet) where {T<:Union{JuMP.Variable,JuMP.AffExpr}}
    """
    Desired output property: the score for “COC” is minimal
    """
    return get_minimality_objective(v_output, [1])
end

property10 = VerificationProperty(
    set_p10_input_constraints,
    get_p10_objective,
    get_default_warm_start,
)


properties = Dict(
    1 => property1,
    2 => property2,
    3 => property3,
    4 => property4,
    5 => property5,
    6 => property6,
    7 => property7,
    8 => property8,
    9 => property9,
    10 => property10,
)
