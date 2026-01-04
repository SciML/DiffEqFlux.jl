abstract type CNFLayer <: AbstractLuxWrapperLayer{:model} end

"""
    FFJORD(model, tspan, input_dims, args...; ad = nothing, basedist = nothing, kwargs...)

Constructs a continuous-time recurrent neural network, also known as a neural ordinary
differential equation (neural ODE), with fast gradient calculation via adjoints [1] and
specialized for density estimation based on continuous normalizing flows (CNF) [2] with a
stochastic approach [2] for the computation of the trace of the dynamics' jacobian. At a
high level this corresponds to the following steps:

 1. Parameterize the variable of interest x(t) as a function f(z, θ, t) of a base variable
    z(t) with known density p\\_z.
 2. Use the transformation of variables formula to predict the density p\\_x as a function
    of the density p\\_z and the trace of the Jacobian of f.
 3. Choose the parameter θ to minimize a loss function of p\\_x (usually the negative
    likelihood of the data).

After these steps one may use the NN model and the learned θ to predict the density p\\_x
for new values of x.

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    dynamics of the model.
  - `basedist`: Distribution of the base variable. Set to the unit normal by default.
  - `input_dims`: Input Dimensions of the model.
  - `tspan`: The timespan to be solved on.
  - `args`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
  - `ad`: The automatic differentiation method to use for the internal jacobian trace.
    Defaults to `AutoForwardDiff()` if full jacobian needs to be computed, i.e.
    `monte_carlo = false`. Else we use `AutoZygote()`.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.

References:

[1] Pontryagin, Lev Semenovich. Mathematical theory of optimal processes. CRC press, 1987.

[2] Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural ordinary
differential equations." In Proceedings of the 32nd International Conference on Neural
Information Processing Systems, pp. 6572-6583. 2018.

[3] Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud.
"Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv
preprint arXiv:1810.01367 (2018).
"""
@concrete struct FFJORD <: CNFLayer
    model <: AbstractLuxLayer
    basedist <: Union{Nothing, Distribution}
    ad
    input_dims
    tspan
    args
    kwargs
end

function LuxCore.initialstates(rng::AbstractRNG, n::FFJORD)
    return (;
        model = LuxCore.initialstates(rng, n.model),
        regularize = false, monte_carlo = true,
    )
end

function FFJORD(
        model, tspan, input_dims, args...; ad = nothing, basedist = nothing, kwargs...
    )
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    return FFJORD(model, basedist, ad, input_dims, tspan, args, kwargs)
end

@inline function __trace_batched(x::AbstractArray{T, 3}) where {T}
    return mapreduce(tr, vcat, eachslice(x; dims = 3); init = similar(x, 0))
end

@inline __norm_batched(x) = sqrt.(sum(abs2, x; dims = 1:(ndims(x) - 1)))

function __ffjord(
        model::StatefulLuxLayer, u::AbstractArray{T, N}, p, ad = nothing,
        regularize::Bool = false, monte_carlo::Bool = true
    ) where {T, N}
    L = size(u, N - 1)
    z = selectdim(u, N - 1, 1:(L - ifelse(regularize, 3, 1)))
    @set! model.ps = p
    mz = model(z, p)
    @assert size(mz) == size(z)
    if monte_carlo
        ad = ad === nothing ? AutoZygote() : ad
        e = CRC.@ignore_derivatives randn!(similar(mz))
        if ad isa AutoForwardDiff
            @assert !regularize "If `regularize = true`, then use `AutoZygote` instead."
            Je = Lux.jacobian_vector_product(model, AutoForwardDiff(), z, e)
            trace_jac = dropdims(
                sum(
                    batched_matmul(
                        reshape(e, 1, :, size(e, N)), reshape(Je, :, 1, size(Je, N))
                    );
                    dims = (1, 2)
                );
                dims = (1, 2)
            )
        elseif ad isa AutoZygote
            eJ = Lux.vector_jacobian_product(model, AutoZygote(), z, e)
            trace_jac = dropdims(
                sum(
                    batched_matmul(
                        reshape(eJ, 1, :, size(eJ, N)), reshape(e, :, 1, size(e, N))
                    );
                    dims = (1, 2)
                );
                dims = (1, 2)
            )
        else
            error("`ad` must be `nothing` or `AutoForwardDiff` or `AutoZygote`.")
        end
        trace_jac = reshape(trace_jac, ntuple(i -> 1, N - 1)..., :)
    else # We can use the batched jacobian since we only care about the trace
        ad = ad === nothing ? AutoForwardDiff() : ad
        if ad isa AutoForwardDiff || ad isa AutoZygote
            J = Lux.batched_jacobian(model, ad, z)
            trace_jac = reshape(__trace_batched(J), ntuple(i -> 1, N - 1)..., :)
            e = CRC.@ignore_derivatives randn!(similar(mz))
            eJ = reshape(batched_matmul(reshape(e, 1, :, size(e, N)), J), size(z))
        else
            error("`ad` must be `nothing` or `AutoForwardDiff` or `AutoZygote`.")
        end
    end
    if regularize
        return cat(mz, -trace_jac, sum(abs2, mz; dims = 1:(N - 1)), __norm_batched(eJ); dims = Val(N - 1))
    else
        return cat(mz, -trace_jac; dims = Val(N - 1))
    end
end

(n::FFJORD)(x, ps, st) = __forward_ffjord(n, x, ps, st)

function __forward_ffjord(n::FFJORD, x::AbstractArray{T, N}, ps, st) where {T, N}
    S = size(x)
    (; regularize, monte_carlo) = st
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())

    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st.model)

    ffjord(u, p, t) = __ffjord(model, u, p, n.ad, regularize, monte_carlo)

    _z = ChainRulesCore.@ignore_derivatives fill!(
        similar(x, S[1:(N - 2)]..., ifelse(regularize, 3, 1), S[N]), zero(T)
    )

    prob = ODEProblem{false}(ffjord, cat(x, _z; dims = Val(N - 1)), n.tspan, ps)
    sol = solve(
        prob, n.args...; sensealg, n.kwargs...,
        save_everystep = false, save_start = false, save_end = true
    )
    pred = __get_pred(sol)
    L = size(pred, N - 1)

    z = selectdim(pred, N - 1, 1:(L - ifelse(regularize, 3, 1)))
    i₁ = L - ifelse(regularize, 2, 0)
    delta_logp = selectdim(pred, N - 1, i₁:i₁)
    if regularize
        λ₁ = selectdim(pred, N, (L - 1):(L - 1))
        λ₂ = selectdim(pred, N, L:L)
    else # For Type Stability
        λ₁ = λ₂ = delta_logp
    end

    if n.basedist === nothing
        logpz = -sum(abs2, z; dims = 1:(N - 1)) / T(2) .- T(prod(S[1:(N - 1)]) / 2 * log(2π))
    else
        logpz = logpdf(n.basedist, z)
    end
    logpx = reshape(logpz, 1, S[N]) .- delta_logp
    return (logpx, λ₁, λ₂), (; model = model.st, regularize, monte_carlo)
end

__get_pred(sol::ODESolution) = last(sol.u)
__get_pred(sol::AbstractArray{T, N}) where {T, N} = selectdim(sol, N, size(sol, N))

function __backward_ffjord(::Type{T1}, n::FFJORD, n_samples::Int, ps, st, rng) where {T1}
    px = n.basedist

    if px === nothing
        x = rng === nothing ? randn(T1, (n.input_dims..., n_samples)) :
            randn(rng, T1, (n.input_dims..., n_samples))
    else
        x = rng === nothing ? rand(px, n_samples) : rand(rng, px, n_samples)
    end

    N, S, T = ndims(x), size(x), eltype(x)
    (; regularize, monte_carlo) = st
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())

    model = StatefulLuxLayer{true}(n.model, nothing, st.model)

    ffjord(u, p, t) = __ffjord(model, u, p, n.ad, regularize, monte_carlo)

    _z = ChainRulesCore.@ignore_derivatives fill!(
        similar(x, S[1:(N - 2)]..., ifelse(regularize, 3, 1), S[N]), zero(T)
    )

    prob = ODEProblem{false}(ffjord, cat(x, _z; dims = Val(N - 1)), reverse(n.tspan), ps)
    sol = solve(
        prob, n.args...; sensealg, n.kwargs...,
        save_everystep = false, save_start = false, save_end = true
    )
    pred = __get_pred(sol)
    L = size(pred, N - 1)

    return selectdim(pred, N - 1, 1:(L - ifelse(regularize, 3, 1)))
end

"""
FFJORD can be used as a distribution to generate new samples by `rand` or estimate densities
by `pdf` or `logpdf` (from `Distributions.jl`).

Arguments:

  - `model`: A FFJORD instance.
  - `regularize`: Whether we use regularization (default: `false`).
  - `monte_carlo`: Whether we use monte carlo (default: `true`).
"""
@concrete struct FFJORDDistribution <: ContinuousMultivariateDistribution
    model <: FFJORD
    ps
    st
end

Base.length(d::FFJORDDistribution) = prod(d.model.input_dims)
Base.eltype(d::FFJORDDistribution) = Lux.recursive_eltype(d.ps)

function Distributions._logpdf(d::FFJORDDistribution, x::AbstractVector)
    return first(first(__forward_ffjord(d.model, reshape(x, :, 1), d.ps, d.st)))
end
function Distributions._logpdf(d::FFJORDDistribution, x::AbstractArray)
    return first(first(__forward_ffjord(d.model, x, d.ps, d.st)))
end
function Distributions._rand!(
        rng::AbstractRNG, d::FFJORDDistribution, x::AbstractArray{<:Real}
    )
    copyto!(x, __backward_ffjord(eltype(d), d.model, size(x, ndims(x)), d.ps, d.st, rng))
    return x
end
