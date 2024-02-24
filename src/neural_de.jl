abstract type NeuralDELayer <: AbstractExplicitContainerLayer{(:model,)} end
abstract type NeuralSDELayer <: AbstractExplicitContainerLayer{(:drift, :diffusion)} end

basic_tgrad(u, p, t) = zero(u)
basic_dde_tgrad(u, h, p, t) = zero(u)

"""
    NeuralODE(model, tspan, alg = nothing, args...; kwargs...)

Constructs a continuous-time recurrant neural network, also known as a neural
ordinary differential equation (neural ODE), with a fast gradient calculation
via adjoints [1]. At a high level this corresponds to solving the forward
differential equation, using a second differential equation that propagates the
derivatives of the loss backwards in time.

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the ̇x.
  - `tspan`: The timespan to be solved on.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropogation.
    Defaults to an adjoint method. See
    the [Local Sensitivity Analysis](https://docs.sciml.ai/SciMLSensitivity/stable/)
    documentation for more details.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.

References:

[1] Pontryagin, Lev Semenovich. Mathematical theory of optimal processes. CRC press, 1987.
"""
@concrete struct NeuralODE{M <: AbstractExplicitLayer} <: NeuralDELayer
    model::M
    tspan
    args
    kwargs
end

function NeuralODE(model, tspan, args...; kwargs...)
    !(model isa AbstractExplicitLayer) && (model = Lux.transform(model))
    return NeuralODE(model, tspan, args, kwargs)
end

function (n::NeuralODE)(x, p, st)
    model = StatefulLuxLayer(n.model, nothing, st)

    dudt(u, p, t) = model(u, p)
    ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)

    return (
        solve(prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), n.kwargs...),
        model.st)
end

"""
    NeuralDSDE(drift, diffusion, tspan, alg = nothing, args...; sensealg = TrackerAdjoint(),
        kwargs...)

Constructs a neural stochastic differential equation (neural SDE) with diagonal noise.

Arguments:

  - `drift`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the drift
    function.
  - `diffusion`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the
    diffusion function. Should output a vector of the same size as the input.
  - `tspan`: The timespan to be solved on.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropogation.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralDSDE{M1 <: AbstractExplicitLayer, M2 <: AbstractExplicitLayer} <:
                 NeuralSDELayer
    drift::M1
    diffusion::M2
    tspan
    args
    kwargs
end

function NeuralDSDE(drift, diffusion, tspan, args...; kwargs...)
    !(drift isa AbstractExplicitLayer) && (drift = Lux.transform(drift))
    !(diffusion isa AbstractExplicitLayer) && (diffusion = Lux.transform(diffusion))
    return NeuralDSDE(drift, diffusion, tspan, args, kwargs)
end

function (n::NeuralDSDE)(x, p, st)
    drift = StatefulLuxLayer(n.drift, nothing, st.drift)
    diffusion = StatefulLuxLayer(n.diffusion, nothing, st.diffusion)

    dudt(u, p, t) = drift(u, p.drift)
    g(u, p, t) = diffusion(u, p.diffusion)

    ff = SDEFunction{false}(dudt, g; tgrad = basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, n.tspan, p)
    return (solve(prob, n.args...; u0 = x, p, sensealg = TrackerAdjoint(), n.kwargs...),
        (; drift = drift.st, diffusion = diffusion.st))
end

"""
    NeuralSDE(drift, diffusion, tspan, nbrown, alg = nothing, args...;
        sensealg=TrackerAdjoint(),kwargs...)

Constructs a neural stochastic differential equation (neural SDE).

Arguments:

  - `drift`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the drift
    function.
  - `diffusion`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the
    diffusion function. Should output a matrix that is `nbrown x size(x, 1)`.
  - `tspan`: The timespan to be solved on.
  - `nbrown`: The number of Brownian processes.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropogation.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralSDE{M1 <: AbstractExplicitLayer, M2 <: AbstractExplicitLayer} <:
                 NeuralSDELayer
    drift::M1
    diffusion::M2
    tspan
    nbrown::Int
    args
    kwargs
end

function NeuralSDE(drift, diffusion, tspan, nbrown, args...; kwargs...)
    !(drift isa AbstractExplicitLayer) && (drift = Lux.transform(drift))
    !(diffusion isa AbstractExplicitLayer) && (diffusion = Lux.transform(diffusion))
    return NeuralSDE(drift, diffusion, tspan, nbrown, args, kwargs)
end

function (n::NeuralSDE)(x, p, st)
    drift = StatefulLuxLayer(n.drift, p.drift, st.drift)
    diffusion = StatefulLuxLayer(n.diffusion, p.diffusion, st.diffusion)

    dudt(u, p, t) = drift(u, p.drift)
    g(u, p, t) = diffusion(u, p.diffusion)

    noise_rate_prototype = CRC.@ignore_derivatives fill!(similar(x, length(x), n.nbrown), 0)

    ff = SDEFunction{false}(dudt, g; tgrad = basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, n.tspan, p; noise_rate_prototype)
    return (solve(prob, n.args...; u0 = x, p, sensealg = TrackerAdjoint(), n.kwargs...),
        (; drift = drift.st, diffusion = diffusion.st))
end

"""
    NeuralCDDE(model, tspan, hist, lags, alg = nothing, args...;
        sensealg = TrackerAdjoint(), kwargs...)

Constructs a neural delay differential equation (neural DDE) with constant delays.

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the
    derivative function. Should take an input of size `[x; x(t - lag_1); ...; x(t - lag_n)]`
    and produce and output shaped like `x`.
  - `tspan`: The timespan to be solved on.
  - `hist`: Defines the history function `h(u, p, t)` for values before the start of the
    integration. Note that `u` is supposed to be used to return a value that matches the size
    of `u`.
  - `lags`: Defines the lagged values that should be utilized in the neural network.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropogation.
    Defaults to using reverse-mode automatic differentiation via Tracker.jl
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralCDDE{M <: AbstractExplicitLayer} <: NeuralDELayer
    model::M
    tspan
    hist
    lags
    args
    kwargs
end

function NeuralCDDE(model, tspan, hist, lags, args...; kwargs...)
    !(model isa AbstractExplicitLayer) && (model = Lux.transform(model))
    return NeuralCDDE(model, tspan, hist, lags, args, kwargs)
end

function (n::NeuralCDDE)(x, ps, st)
    model = StatefulLuxLayer(n.model, nothing, st)

    function dudt(u, h, p, t)
        xs = mapfoldl(lag -> h(p, t - lag), vcat, n.lags)
        return model(vcat(u, xs), p)
    end

    ff = DDEFunction{false}(dudt; tgrad = basic_dde_tgrad)
    prob = DDEProblem{false}(ff, x, (p, t) -> n.hist(x, p, t), n.tspan, ps;
        constant_lags = n.lags)

    return (solve(prob, n.args...; sensealg = TrackerAdjoint(), n.kwargs...), model.st)
end

"""
    NeuralDAE(model, constraints_model, tspan, args...; differential_vars = nothing,
        sensealg = TrackerAdjoint(), kwargs...)

Constructs a neural differential-algebraic equation (neural DAE).

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the
    derivative function. Should take an input of size `x` and produce the residual of
    `f(dx,x,t)` for only the differential variables.
  - `constraints_model`: A function `constraints_model(u,p,t)` for the fixed
    constraints to impose on the algebraic equations.
  - `tspan`: The timespan to be solved on.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropogation.
    Defaults to using reverse-mode automatic differentiation via Tracker.jl
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralDAE{M <: AbstractExplicitLayer} <: NeuralDELayer
    model::M
    constraints_model
    tspan
    args
    differential_vars
    kwargs
end

function NeuralDAE(model, constraints_model, tspan, args...; differential_vars = nothing,
        kwargs...)
    !(model isa AbstractExplicitLayer) && (model = Lux.transform(model))
    return NeuralDAE(model, constraints_model, tspan, args, differential_vars, kwargs)
end

function (n::NeuralDAE)(u_du::Tuple, p, st)
    u0, du0 = u_du
    model = StatefulLuxLayer(n.model, nothing, st)

    function f(du, u, p, t)
        nn_out = model(vcat(u, du), p)
        alg_out = n.constraints_model(u, p, t)
        iter_nn, iter_const = 0, 0
        res = map(n.differential_vars) do isdiff
            if isdiff
                iter_nn += 1
                nn_out[iter_nn]
            else
                iter_const += 1
                alg_out[iter_const]
            end
        end
        return res
    end

    prob = DAEProblem{false}(f, du0, u0, n.tspan, p; n.differential_vars)
    return solve(prob, n.args...; sensealg = TrackerAdjoint(), n.kwargs...), st
end

"""
    NeuralODEMM(model, constraints_model, tspan, mass_matrix, alg = nothing, args...;
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()), kwargs...)

Constructs a physically-constrained continuous-time recurrant neural network, also known as
a neural differential-algebraic equation (neural DAE), with a mass matrix and a fast
gradient calculation via adjoints [1]. The mass matrix formulation is:

```math
Mu' = f(u,p,t)
```

where `M` is semi-explicit, i.e. singular with zeros for rows corresponding to the
constraint equations.

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that defines the ̇`f(u,p,t)`
  - `constraints_model`: A function `constraints_model(u,p,t)` for the fixed constraints to
    impose on the algebraic equations.
  - `tspan`: The timespan to be solved on.
  - `mass_matrix`: The mass matrix associated with the DAE.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the default
    algorithm from DifferentialEquations.jl. This method requires an implicit ODE solver
    compatible with singular mass matrices. Consult the
    [DAE solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/) documentation
    for more details.
  - `sensealg`: The choice of differentiation algorithm used in the backpropogation.
    Defaults to an adjoint method. See
    the [Local Sensitivity Analysis](https://docs.sciml.ai/SciMLSensitivity/stable/)
    documentation for more details.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralODEMM{M <: AbstractExplicitLayer} <: NeuralDELayer
    model::M
    constraints_model
    tspan
    mass_matrix
    args
    kwargs
end

function NeuralODEMM(model, constraints_model, tspan, mass_matrix, args...; kwargs...)
    !(model isa AbstractExplicitLayer) && (model = Lux.transform(model))
    return NeuralODEMM(model, constraints_model, tspan, mass_matrix, args, kwargs)
end

function (n::NeuralODEMM)(x, ps, st)
    model = StatefulLuxLayer(n.model, nothing, st)

    function f(u, p, t)
        nn_out = model(u, p)
        alg_out = n.constraints_model(u, p, t)
        return vcat(nn_out, alg_out)
    end

    dudt = ODEFunction{false}(f; mass_matrix = n.mass_matrix, tgrad = basic_tgrad)
    prob = ODEProblem{false}(dudt, x, n.tspan, ps)

    return (
        solve(prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), n.kwargs...),
        model.st)
end

"""
    AugmentedNDELayer(nde, adim::Int)

Constructs an Augmented Neural Differential Equation Layer.

Arguments:

  - `nde`: Any Neural Differential Equation Layer.
  - `adim`: The number of dimensions the initial conditions should be lifted.

References:

[1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural ODEs." In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3140-3150. 2019.
"""
function AugmentedNDELayer(model::Union{NeuralDELayer, NeuralSDELayer}, adim::Int)
    return Chain(Base.Fix2(__augment, adim), model)
end

function __augment(x::AbstractVector, augment_dim::Int)
    y = CRC.@ignore_derivatives fill!(similar(x, augment_dim), 0)
    return vcat(x, y)
end

function __augment(x::AbstractArray, augment_dim::Int)
    y = CRC.@ignore_derivatives fill!(
        similar(x, size(x)[1:(ndims(x) - 2)]...,
            augment_dim, size(x, ndims(x))), 0)
    return cat(x, y; dims = Val(ndims(x) - 1))
end

"""
    DimMover(from, to)

Constructs a Dimension Mover Layer.

We can have Flux's conventional order `(data, channel, batch)` by using it as the last layer
of `Flux.Chain` to swap the batch-index and the time-index of the Neural DE's output
considering that each time point is a channel.
"""
@concrete struct DimMover <: AbstractExplicitLayer
    from
    to
end

function DimMover(; from = -2, to = -1)
    @assert from !== 0 && to !== 0
    return DimMover(from, to)
end

function (dm::DimMover)(x, ps, st)
    from = dm.from > 0 ? dm.from : (ndims(x) + 1 + dm.from)
    to = dm.to > 0 ? dm.to : (ndims(x) + 1 + dm.to)

    return cat(eachslice(x; dims = from)...; dims = to), st
end
