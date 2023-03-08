abstract type CNFLayer <: LuxCore.AbstractExplicitContainerLayer{(:model,)} end

"""
Constructs a continuous-time recurrent neural network, also known as a neural
ordinary differential equation (neural ODE), with fast gradient calculation
via adjoints [1] and specialized for density estimation based on continuous
normalizing flows (CNF) [2] with a stochastic approach [2] for the computation of the trace
of the dynamics' jacobian. At a high level this corresponds to the following steps:

1. Parameterize the variable of interest x(t) as a function f(z, θ, t) of a base variable z(t) with known density p_z;
2. Use the transformation of variables formula to predict the density p_x as a function of the density p_z and the trace of the Jacobian of f;
3. Choose the parameter θ to minimize a loss function of p_x (usually the negative likelihood of the data);

After these steps one may use the NN model and the learned θ to predict the density p_x for new values of x.

```julia
FFJORD(model, basedist=nothing, monte_carlo=false, tspan, args...; kwargs...)
```
Arguments:
- `model`: A Flux.Chain or Lux.AbstractExplicitLayer neural network that defines the dynamics of the model.
- `basedist`: Distribution of the base variable. Set to the unit normal by default.
- `tspan`: The timespan to be solved on.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
  documentation for more details.

References:

[1] Pontryagin, Lev Semenovich. Mathematical theory of optimal processes. CRC press, 1987.

[2] Chen, Ricky TQ, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. "Neural ordinary differential equations." In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 6572-6583. 2018.

[3] Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).

"""
struct FFJORD{M, P, RE, D, T, A, K} <: CNFLayer where {M, P <: Union{AbstractVector{<: AbstractFloat}, Nothing}, RE <: Union{Function, Nothing}, D <: Distribution, T, A, K}
    model::M
    p::P
    re::RE
    basedist::D
    tspan::T
    args::A
    kwargs::K

    function FFJORD(model::LuxCore.AbstractExplicitLayer,tspan,args...;p=nothing,basedist=nothing,kwargs...)
        re = nothing
        if isnothing(basedist)
            size_input = model.layers.layer_1.in_dims
            type_input = eltype(tspan)
            basedist = MvNormal(zeros(type_input, size_input), Diagonal(ones(type_input, size_input)))
        end
        new{typeof(model),typeof(p),typeof(re),
          typeof(basedist),typeof(tspan),typeof(args),typeof(kwargs)}(
          model,p,re,basedist,tspan,args,kwargs)
    end
end

_norm_batched(x::AbstractMatrix) = sqrt.(sum(x.^2, dims=1))

function jacobian_fn(f, x::AbstractVector, args...)
    y::AbstractVector, back = Zygote.pullback(f, x)
    ȳ(i) = [i == j for j = 1:length(y)]
    vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
end

function jacobian_fn(f::LuxCore.AbstractExplicitLayer, x::AbstractMatrix, args...)
    p,st = args
    y, back = Zygote.pullback((z,ps,s)->f(z,ps,s)[1], x, p, st)
    z = ChainRulesCore.@ignore_derivatives similar(y)
    ChainRulesCore.@ignore_derivatives fill!(z, zero(eltype(x)))
    vec = Zygote.Buffer(x, size(x, 1), size(x, 1), size(x, 2))
    for i in 1:size(y, 1)
        ChainRulesCore.@ignore_derivatives z[i, :] .= one(eltype(x))
        vec[i, :, :] .= back(z)[1]
        ChainRulesCore.@ignore_derivatives z[i, :] .= zero(eltype(x))
    end
    copy(vec)
end

_trace_batched(x::AbstractArray{T, 3}) where T =
    reshape([tr(x[:, :, i]) for i in 1:size(x, 3)], 1, size(x, 3))

function ffjord(u, p, t, re::LuxCore.AbstractExplicitLayer, e, st;
    regularize=false, monte_carlo=true)
    if regularize
        z = u[1:end - 3, :]
        if monte_carlo
            mz, back = Zygote.pullback((x,ps,s)->re(x,ps,s)[1], z, p, st)
            eJ = back(e)[1]
            trace_jac = sum(eJ .* e, dims=1)
        else
            mz = re(z, p, st)[1]
            trace_jac = _trace_batched(jacobian_fn(re, z, p, st))
        end
        vcat(mz, -trace_jac, sum(abs2, mz, dims=1), _norm_batched(eJ))
    else
        z = u[1:end - 1, :]
        if monte_carlo
            mz, back = Zygote.pullback((x,ps,s)->re(x,ps,s)[1], z, p, st)
            eJ = back(e)[1]
            trace_jac = sum(eJ .* e, dims=1)
        else
            mz = re(z, p, st)[1]
            trace_jac = _trace_batched(jacobian_fn(re, z, p, st))
        end
        vcat(mz, -trace_jac)
    end
end

# When running on GPU e needs to be passed separately, when using Lux pass st as a kwarg
(n::FFJORD)(args...; kwargs...) = forward_ffjord(n, args...; kwargs...)

function forward_ffjord(n::FFJORD, x, p=n.p, e=randn(eltype(x), size(x));
                        regularize=false, monte_carlo=true, st=nothing)
    pz = n.basedist
    sensealg = InterpolatingAdjoint()
    ffjord_(u, p, t) = ffjord(u, p, t, n.re, e, st; regularize, monte_carlo)
    # ffjord_(u, p, t) = ffjord(u, p, t, n.re, e; regularize, monte_carlo)
    if regularize
        _z = ChainRulesCore.@ignore_derivatives similar(x, 3, size(x, 2))
        ChainRulesCore.@ignore_derivatives fill!(_z, zero(eltype(x)))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), n.tspan, p)
        pred = solve(prob, n.args...; sensealg, n.kwargs...)[:, :, end]
        z = pred[1:end - 3, :]
        delta_logp = pred[end - 2:end - 2, :]
        λ₁ = pred[end - 1, :]
        λ₂ = pred[end, :]
    else
        _z = ChainRulesCore.@ignore_derivatives similar(x, 1, size(x, 2))
        ChainRulesCore.@ignore_derivatives fill!(_z, zero(eltype(x)))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), n.tspan, p)
        pred = solve(prob, n.args...; sensealg, n.kwargs...)[:, :, end]
        z = pred[1:end - 1, :]
        delta_logp = pred[end:end, :]
        λ₁ = λ₂ = _z[1, :]
    end

    logpz = reshape(logpdf(pz, z), 1, size(x, 2))
    logpx = logpz .- delta_logp

    logpx, λ₁, λ₂
end

function backward_ffjord(n::FFJORD, n_samples, p=n.p, e=randn(eltype(n.model[1].weight), n_samples);
                         regularize=false, monte_carlo=true, rng=nothing, st=nothing)
    px = n.basedist
    x = isnothing(rng) ? rand(px, n_samples) : rand(rng, px, n_samples)
    sensealg = InterpolatingAdjoint()
    ffjord_(u, p, t) = ffjord(u, p, t, n.re, e, st; regularize, monte_carlo)
    if regularize
        _z = ChainRulesCore.@ignore_derivatives similar(x, 3, size(x, 2))
        ChainRulesCore.@ignore_derivatives fill!(_z, zero(eltype(x)))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), reverse(n.tspan), p)
        pred = solve(prob, n.args...; sensealg, n.kwargs...)[:, :, end]
        z = pred[1:end - 3, :]
    else
        _z = ChainRulesCore.@ignore_derivatives similar(x, 1, size(x, 2))
        ChainRulesCore.@ignore_derivatives fill!(_z, zero(eltype(x)))
        prob = ODEProblem{false}(ffjord_, vcat(x, _z), reverse(n.tspan), p)
        pred = solve(prob, n.args...; sensealg, n.kwargs...)[:, :, end]
        z = pred[1:end - 1, :]
    end

    z
end

"""
FFJORD can be used as a distribution to generate new samples by `rand` or estimate densities by `pdf` or `logpdf` (from `Distributions.jl`).

Arguments:
- `model`: A FFJORD instance
- `regularize`: Whether we use regularization (default: `false`)
- `monte_carlo`: Whether we use monte carlo (default: `true`)

"""
struct FFJORDDistribution <: ContinuousMultivariateDistribution
    model::FFJORD
    regularize::Bool
    monte_carlo::Bool
end

FFJORDDistribution(model; regularize=false, monte_carlo=true) = FFJORDDistribution(model, regularize, monte_carlo)

Base.length(d::FFJORDDistribution) = size(d.model.model[1].weight, 2)
Base.eltype(d::FFJORDDistribution) = eltype(d.model.model[1].weight)
Distributions._logpdf(d::FFJORDDistribution, x::AbstractArray) = forward_ffjord(d.model, x; d.regularize, d.monte_carlo)[1]
Distributions._rand!(rng::AbstractRNG, d::FFJORDDistribution, x::AbstractVector{<: Real}) = (x[:] = backward_ffjord(d.model, size(x, 2); d.regularize, d.monte_carlo, rng))
Distributions._rand!(rng::AbstractRNG, d::FFJORDDistribution, A::DenseMatrix{<: Real}) = (A[:] = backward_ffjord(d.model, size(A, 2); d.regularize, d.monte_carlo, rng))
