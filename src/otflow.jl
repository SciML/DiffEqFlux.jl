# Abstract type for CNF layers
abstract type CNFLayer <: AbstractLuxWrapperLayer{:model} end

"""
    OTFlow(model, tspan, input_dims, args...; ad = nothing, basedist = nothing, kwargs...)

Constructs a continuous-time neural network based on optimal transport (OT) theory, using
a potential function to define the dynamics and exact trace computation for the Jacobian.
This is a continuous normalizing flow (CNF) model specialized for density estimation.

Arguments:
  - `model`: A `Lux.AbstractLuxLayer` neural network that defines the potential function Φ.
  - `basedist`: Distribution of the base variable. Set to the unit normal by default.
  - `input_dims`: Input dimensions of the model.
  - `tspan`: The timespan to be solved on.
  - `args`: Additional arguments splatted to the ODE solver.
  - `ad`: The automatic differentiation method to use for the internal Jacobian trace.
  - `kwargs`: Additional arguments splatted to the ODE solver.
"""
@concrete struct OTFlow <: CNFLayer
    model <: AbstractLuxLayer
    basedist <: Union{Nothing, Distribution}
    ad
    input_dims
    tspan
    args
    kwargs
end

function LuxCore.initialstates(rng::AbstractRNG, n::OTFlow)
    # Initialize the model's state and other parameters
    model_st = LuxCore.initialstates(rng, n.model)
    return (; model = model_st, regularize = false)
end

function OTFlow(
    model, tspan, input_dims, args...; ad = nothing, basedist = nothing, kwargs...
)
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    return OTFlow(model, basedist, ad, input_dims, tspan, args, kwargs)
end

# Dynamics function for OTFlow
function __otflow_dynamics(model::StatefulLuxLayer, u::AbstractArray{T, N}, p, ad = nothing) where {T, N}
    L = size(u, N - 1)
    z = selectdim(u, N - 1, 1:(L - 1))  # Extract the state variables
    @set! model.ps = p

    # Compute the potential function Φ(z)
    Φ = model(z, p)

    # Compute the gradient of Φ(z) to get the dynamics v(z) = -∇Φ(z)
    ∇Φ = gradient(z -> sum(model(z, p)), z)[1]
    v = -∇Φ

    # Compute the trace of the Jacobian of the dynamics (∇v)
    H = Zygote.hessian(z -> sum(model(z, p)), z)
    trace_jac = tr(H)

    # Return the dynamics and the trace term
    return cat(v, -reshape(trace_jac, ntuple(i -> 1, N - 1)..., :); dims = Val(N - 1))
end

# Forward pass for OTFlow
function (n::OTFlow)(x, ps, st)
    return __forward_otflow(n, x, ps, st)
end

function __forward_otflow(n::OTFlow, x::AbstractArray{T, N}, ps, st) where {T, N}
    S = size(x)
    (; regularize) = st
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())

    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st.model)

    otflow_dynamics(u, p, t) = __otflow_dynamics(model, u, p, n.ad)

    _z = ChainRulesCore.@ignore_derivatives fill!(
        similar(x, S[1:(N - 2)]..., 1, S[N]), zero(T))

    prob = ODEProblem{false}(otflow_dynamics, cat(x, _z; dims = Val(N - 1)), n.tspan, ps)
    sol = solve(prob, n.args...; sensealg, n.kwargs...,
        save_everystep = false, save_start = false, save_end = true)
    pred = __get_pred(sol)
    L = size(pred, N - 1)

    z = selectdim(pred, N - 1, 1:(L - 1))
    delta_logp = selectdim(pred, N - 1, L:L)

    if n.basedist === nothing
        logpz = -sum(abs2, z; dims = 1:(N - 1)) / T(2) .-
                T(prod(S[1:(N - 1)]) / 2 * log(2π))
    else
        logpz = logpdf(n.basedist, z)
    end
    logpx = reshape(logpz, 1, S[N]) .- delta_logp
    return (logpx,), (; model = model.st, regularize)
end

# Backward pass for OTFlow
function __backward_otflow(::Type{T1}, n::OTFlow, n_samples::Int, ps, st, rng) where {T1}
    px = n.basedist

    if px === nothing
        x = rng === nothing ? randn(T1, (n.input_dims..., n_samples)) :
            randn(rng, T1, (n.input_dims..., n_samples))
    else
        x = rng === nothing ? rand(px, n_samples) : rand(rng, px, n_samples)
    end

    N, S, T = ndims(x), size(x), eltype(x)
    (; regularize) = st
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())

    model = StatefulLuxLayer{true}(n.model, nothing, st.model)

    otflow_dynamics(u, p, t) = __otflow_dynamics(model, u, p, n.ad)

    _z = ChainRulesCore.@ignore_derivatives fill!(
        similar(x, S[1:(N - 2)]..., 1, S[N]), zero(T))

    prob = ODEProblem{false}(otflow_dynamics, cat(x, _z; dims = Val(N - 1)), reverse(n.tspan), ps)
    sol = solve(prob, n.args...; sensealg, n.kwargs...,
        save_everystep = false, save_start = false, save_end = true)
    pred = __get_pred(sol)
    L = size(pred, N - 1)

    return selectdim(pred, N - 1, 1:(L - 1))
end

# OTFlow can be used as a distribution
@concrete struct OTFlowDistribution <: ContinuousMultivariateDistribution
    model <: OTFlow
    ps
    st
end

Base.length(d::OTFlowDistribution) = prod(d.model.input_dims)
Base.eltype(d::OTFlowDistribution) = Lux.recursive_eltype(d.ps)

function Distributions._logpdf(d::OTFlowDistribution, x::AbstractVector)
    return first(first(__forward_otflow(d.model, reshape(x, :, 1), d.ps, d.st)))
end
function Distributions._logpdf(d::OTFlowDistribution, x::AbstractArray)
    return first(first(__forward_otflow(d.model, x, d.ps, d.st)))
end
function Distributions._rand!(
    rng::AbstractRNG, d::OTFlowDistribution, x::AbstractArray{<:Real}
)
    copyto!(x, __backward_otflow(eltype(d), d.model, size(x, ndims(x)), d.ps, d.st, rng))
    return x
end