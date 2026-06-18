abstract type NeuralDELayer <: AbstractLuxWrapperLayer{:model} end
abstract type NeuralSDELayer <: AbstractLuxContainerLayer{(:drift, :diffusion)} end

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

  - `model`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    ̇x.
  - `tspan`: The timespan to be solved on.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropagation.
    Defaults to an adjoint method. See
    the [Local Sensitivity Analysis](https://docs.sciml.ai/SciMLSensitivity/stable/)
    documentation for more details.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.

References:

[1] Pontryagin, Lev Semenovich. Mathematical theory of optimal processes. CRC press, 1987.
"""
@concrete struct NeuralODE <: NeuralDELayer
    model <: AbstractLuxLayer
    tspan
    args
    kwargs
end

function NeuralODE(model, tspan, args...; kwargs...)
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    return NeuralODE(model, tspan, args, kwargs)
end

function (n::NeuralODE)(x, p, st)
    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st)

    dudt(u, p, t) = model(u, p)
    ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)

    return (
        solve(
            prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), n.kwargs...
        ),
        model.st,
    )
end

"""
    NeuralDSDE(drift, diffusion, tspan, alg = nothing, args...; sensealg = TrackerAdjoint(),
        kwargs...)

Constructs a neural stochastic differential equation (neural SDE) with diagonal noise.

Arguments:

  - `drift`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    drift function.
  - `diffusion`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines
    the diffusion function. Should output a vector of the same size as the input.
  - `tspan`: The timespan to be solved on.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropagation.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralDSDE <: NeuralSDELayer
    drift <: AbstractLuxLayer
    diffusion <: AbstractLuxLayer
    tspan
    args
    kwargs
end

function NeuralDSDE(drift, diffusion, tspan, args...; kwargs...)
    !(drift isa AbstractLuxLayer) && (drift = FromFluxAdaptor()(drift))
    !(diffusion isa AbstractLuxLayer) && (diffusion = FromFluxAdaptor()(diffusion))
    return NeuralDSDE(drift, diffusion, tspan, args, kwargs)
end

function (n::NeuralDSDE)(x, p, st)
    drift = StatefulLuxLayer{fixed_state_type(n.drift)}(n.drift, nothing, st.drift)
    diffusion = StatefulLuxLayer{fixed_state_type(n.diffusion)}(
        n.diffusion, nothing, st.diffusion
    )

    dudt(u, p, t) = drift(u, p.drift)
    g(u, p, t) = diffusion(u, p.diffusion)

    ff = SDEFunction{false}(dudt, g; tgrad = basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, n.tspan, p)
    return (
        solve(prob, n.args...; u0 = x, p, sensealg = TrackerAdjoint(), n.kwargs...),
        (; drift = drift.st, diffusion = diffusion.st),
    )
end

"""
    NeuralSDE(drift, diffusion, tspan, nbrown, alg = nothing, args...;
        sensealg=TrackerAdjoint(), kwargs...)

Constructs a neural stochastic differential equation (neural SDE).

Arguments:

  - `drift`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    drift function.
  - `diffusion`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines
    the diffusion function. Should output a matrix that is `nbrown x size(x, 1)`.
  - `tspan`: The timespan to be solved on.
  - `nbrown`: The number of Brownian processes.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropagation.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralSDE <: NeuralSDELayer
    drift <: AbstractLuxLayer
    diffusion <: AbstractLuxLayer
    tspan
    nbrown::Int
    args
    kwargs
end

function NeuralSDE(drift, diffusion, tspan, nbrown, args...; kwargs...)
    !(drift isa AbstractLuxLayer) && (drift = FromFluxAdaptor()(drift))
    !(diffusion isa AbstractLuxLayer) && (diffusion = FromFluxAdaptor()(diffusion))
    return NeuralSDE(drift, diffusion, tspan, nbrown, args, kwargs)
end

function (n::NeuralSDE)(x, p, st)
    drift = StatefulLuxLayer{fixed_state_type(n.drift)}(n.drift, p.drift, st.drift)
    diffusion = StatefulLuxLayer{fixed_state_type(n.diffusion)}(
        n.diffusion, p.diffusion, st.diffusion
    )

    dudt(u, p, t) = drift(u, p.drift)
    g(u, p, t) = diffusion(u, p.diffusion)

    noise_rate_prototype = CRC.@ignore_derivatives fill!(similar(x, length(x), n.nbrown), 0)

    ff = SDEFunction{false}(dudt, g; tgrad = basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, n.tspan, p; noise_rate_prototype)
    return (
        solve(prob, n.args...; u0 = x, p, sensealg = TrackerAdjoint(), n.kwargs...),
        (; drift = drift.st, diffusion = diffusion.st),
    )
end

"""
    NeuralCDDE(model, tspan, hist, lags, alg = nothing, args...;
        sensealg = TrackerAdjoint(), kwargs...)

Constructs a neural delay differential equation (neural DDE) with constant delays.

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    derivative function. Should take an input of size `[x; x(t - lag_1); ...; x(t - lag_n)]`
    and produce and output shaped like `x`.
  - `tspan`: The timespan to be solved on.
  - `hist`: Defines the history function `h(u, p, t)` for values before the start of the
    integration. Note that `u` is supposed to be used to return a value that matches the
    size of `u`.
  - `lags`: Defines the lagged values that should be utilized in the neural network.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropagation.
    Defaults to using reverse-mode automatic differentiation via Tracker.jl
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralCDDE <: NeuralDELayer
    model <: AbstractLuxLayer
    tspan
    hist
    lags
    args
    kwargs
end

function NeuralCDDE(model, tspan, hist, lags, args...; kwargs...)
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    return NeuralCDDE(model, tspan, hist, lags, args, kwargs)
end

function (n::NeuralCDDE)(x, ps, st)
    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st)

    function dudt(u, h, p, t)
        xs = mapfoldl(lag -> h(p, t - lag), vcat, n.lags)
        return model(vcat(u, xs), p)
    end

    ff = DDEFunction{false}(dudt; tgrad = basic_dde_tgrad)
    prob = DDEProblem{false}(
        ff, x, (p, t) -> n.hist(x, p, t), n.tspan, ps; constant_lags = n.lags
    )

    return (solve(prob, n.args...; sensealg = TrackerAdjoint(), n.kwargs...), model.st)
end

"""
    NeuralDAE(model, constraints_model, tspan, args...; differential_vars = nothing,
        sensealg = TrackerAdjoint(), kwargs...)

Constructs a neural differential-algebraic equation (neural DAE).

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    derivative function. Should take an input of size `x` and produce the residual of
    `f(dx,x,t)` for only the differential variables.
  - `constraints_model`: A function `constraints_model(u,p,t)` for the fixed
    constraints to impose on the algebraic equations.
  - `tspan`: The timespan to be solved on.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
    default algorithm from DifferentialEquations.jl.
  - `sensealg`: The choice of differentiation algorithm used in the backpropagation.
    Defaults to using reverse-mode automatic differentiation via Tracker.jl
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralDAE <: NeuralDELayer
    model <: AbstractLuxLayer
    constraints_model
    tspan
    args
    differential_vars
    kwargs
end

function NeuralDAE(
        model, constraints_model, tspan, args...; differential_vars = nothing, kwargs...
    )
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    return NeuralDAE(model, constraints_model, tspan, args, differential_vars, kwargs)
end

function (n::NeuralDAE)(u_du::Tuple, p, st)
    u0, du0 = u_du
    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st)

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

  - `model`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    ̇`f(u,p,t)`
  - `constraints_model`: A function `constraints_model(u,p,t)` for the fixed constraints to
    impose on the algebraic equations.
  - `tspan`: The timespan to be solved on.
  - `mass_matrix`: The mass matrix associated with the DAE.
  - `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the default
    algorithm from DifferentialEquations.jl. This method requires an implicit ODE solver
    compatible with singular mass matrices. Consult the
    [DAE solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/) documentation
    for more details.
  - `sensealg`: The choice of differentiation algorithm used in the backpropagation.
    Defaults to an adjoint method. See
    the [Local Sensitivity Analysis](https://docs.sciml.ai/SciMLSensitivity/stable/)
    documentation for more details.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
@concrete struct NeuralODEMM <: NeuralDELayer
    model <: AbstractLuxLayer
    constraints_model
    tspan
    mass_matrix
    args
    kwargs
end

function NeuralODEMM(model, constraints_model, tspan, mass_matrix, args...; kwargs...)
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    return NeuralODEMM(model, constraints_model, tspan, mass_matrix, args, kwargs)
end

function (n::NeuralODEMM)(x, ps, st)
    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st)

    function f(u, p, t)
        nn_out = model(u, p)
        alg_out = n.constraints_model(u, p, t)
        return vcat(nn_out, alg_out)
    end

    dudt = ODEFunction{false}(f; mass_matrix = n.mass_matrix, tgrad = basic_tgrad)
    prob = ODEProblem{false}(dudt, x, n.tspan, ps)

    return (
        solve(
            prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), n.kwargs...
        ),
        model.st,
    )
end

"""
    AugmentedNDELayer(nde, adim::Int)

Constructs an Augmented Neural Differential Equation Layer.

Arguments:

  - `nde`: Any Neural Differential Equation Layer.
  - `adim`: The number of dimensions the initial conditions should be lifted.

References:

[1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural ODEs." In
Proceedings of the 33rd International Conference on Neural Information Processing
Systems, pp. 3140-3150. 2019.
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
        similar(x, size(x)[1:(ndims(x) - 2)]..., augment_dim, size(x, ndims(x))), 0
    )
    return cat(x, y; dims = Val(ndims(x) - 1))
end

"""
    DimMover(from, to)

Constructs a Dimension Mover Layer.

We can have Lux's conventional order `(data, channel, batch)` by using it as the last layer
of `AbstractLuxLayer` to swap the batch-index and the time-index of the Neural DE's
output considering that each time point is a channel.
"""
@concrete struct DimMover <: AbstractLuxLayer
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


# Introduce a parent type for all Galerkin basis families, for exmaple Fourier bassis, Chebyshev basis.
abstract type AbstractGalerkinBasis end

# Interface function, to be implemented by concrete basis types, and return the number of basis functions.
function basisdim end

# Interface function, given the current solver time `t`, return the vector of basis function values psi(t).
function basis_eval end
function lift_parameter_tree end
function reconstruct_parameter_tree end

# Helper for the constant-only mode index, which is always the first basis function.
constant_mode(::AbstractGalerkinBasis) = 1

"""
    FourierBasis{M}()

A simple Fourier basis with `M` total modes laid out as
    [1, sin(2pi tau), cos(2pi tau), sin(4pi tau), cos(4pi tau), ...]

where `tau` is the normalized solver time on `[0, 1]`.

Notes:
- `M == 1` gives a constant-only basis, which should reduce to an ordinary `NeuralODE`.
- Even `M` values are allowed, though the last mode will be an unmatched sine term.
"""
# Define a concrete basis type. M is the number of basis entries, known at compile time.
struct FourierBasis{M} <: AbstractGalerkinBasis end

# The number of basis functions is just M, but we also check that M is at least 1 to avoid invalid zero-mode bases.
basisdim(::FourierBasis{M}) where {M} = M >= 1 ? M : throw(ArgumentError("FourierBasis{$M} requires M ≥ 1."))

# Evaluate the Fourier basis functions at time t, given the timespan for normalization. The output is a length-M vector of basis values psi(t).
function basis_eval(::FourierBasis{M}, t, tspan) where {M}
    t0, t1 = tspan
    tau = (t - t0) / (t1 - t0)

    return map(1:M) do i
        if i == 1
            one(tau)
        else
            k = i ÷ 2
            iseven(i) ? sinpi(2k * tau) : cospi(2k * tau)
        end
    end
end

"""
    GalerkinNeuralODE(model, basis, tspan, alg = nothing, args...; kwargs...)

Construct a Neural ODE-like layer whose parameters vary continuously with
depth/time via a Galerkin expansion:
    Theta(t) = Sum_j alpha_j * psi_j(t)

where `psi_j` comes from `basis` and the trainable object is the coefficient tree `alpha`.

This implementation reuses the current DiffEqFlux solve-and-adjoint path used by `NeuralODE`, but
reconstructs ordinary model parameters inside the ODE RHS at each solver time.

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the
    ̇x.
  - `tspan`: The timespan to be solved on.
  - `args`: Solver positional arguments, such as Tsit5()
  - `kwargs`: Solver keyword arguments, such as saveat, tolerances, callbacks
"""
@concrete struct GalerkinNeuralODE <: NeuralDELayer
    model <: AbstractLuxLayer
    basis <: AbstractGalerkinBasis
    tspan
    args
    kwargs
end

function GalerkinNeuralODE(model, basis::AbstractGalerkinBasis, tspan, args...; kwargs...)
    # If the user passes a Flux model instead of a Lux model, adapt it to Lux first
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    return GalerkinNeuralODE(model, basis, tspan, args, kwargs)
end

# ------------------------------------------------------------------------------
# Lux interface
# ------------------------------------------------------------------------------

# Lux requires custom layers to implement `initialparameters`, `initialstates`, `parameterlength`, and `statelength` methods
function LuxCore.initialparameters(rng::AbstractRNG, g::GalerkinNeuralODE)
    # Start with the ordinary parameter tree from the base model
    base_ps = LuxCore.initialparameters(rng, g.model)
    # Lift it into the coefficient tree for the Galerkin expansion, which will be trained instead of the base parameters
    return lift_parameter_tree(base_ps, g.basis)
end

# The state tree is unaffected by the Galerkin expansion, so we can just delegate to the base model.
LuxCore.initialstates(rng::AbstractRNG, g::GalerkinNeuralODE) =
    LuxCore.initialstates(rng, g.model)

# The number of trainable parameters is the number of basis functions times the number of parameters in the base model
LuxCore.parameterlength(g::GalerkinNeuralODE) =
    basisdim(g.basis) * LuxCore.parameterlength(g.model)

# The state length is the same as the base model, since the state tree is unchanged.
LuxCore.statelength(g::GalerkinNeuralODE) =
    LuxCore.statelength(g.model)

# ------------------------------------------------------------------------------
# Parameter lifting helpers: ordinary Lux parameter tree -> Galerkin coefficient tree
# ------------------------------------------------------------------------------

# Recursively lift an ordinary parameter tree of NamedTuples into a coefficient tree for the Galerkin expansion.
lift_parameter_tree(ps::NamedTuple, basis::AbstractGalerkinBasis) =
    map(v -> lift_parameter_tree(v, basis), ps)

# Same for Tuple structures.
lift_parameter_tree(ps::Tuple, basis::AbstractGalerkinBasis) =
    map(v -> lift_parameter_tree(v, basis), ps)

# Base case: if the parameter leaf is `nothing`, we can just return `nothing` in the lifted tree, since it won't be trained or used in the RHS.
lift_parameter_tree(::Nothing, ::AbstractGalerkinBasis) = nothing

# Base case: handles array-valued parameter leaves
function lift_parameter_tree(x::AbstractArray, basis::AbstractGalerkinBasis)
    # Number of basis functions
    m = basisdim(basis)

    # Allocate a new array with the same element type and backend as x, but with one extra leading dimension of size m.
    alpha = similar(x, (m, size(x)...))

    # Initialize all coefficient slices to zero.
    fill!(alpha, zero(eltype(alpha)))

    # Pick the constant basis slice and copy the original parameter x into it.
    # So the initial model is exactly the original static-parameter model, with all nonconstant modes turned off.
    selectdim(alpha, 1, constant_mode(basis)) .= x

    # Return the lifted array leaf.
    return alpha
end

# Base case: if the parameter leaf is a scalar number
function lift_parameter_tree(x::Number, basis::AbstractGalerkinBasis)
    # Create a length-m vector of scalar coefficients.
    alpha = fill(zero(x), basisdim(basis))

    # Put the original scalar into the constant mode.
    alpha[constant_mode(basis)] = x

    # Return the lifted scalar leaf.
    return alpha
end

# Fallback method for unsupported leaf types, which throws an error if we encounter a parameter leaf type that we don't know how to lift.
lift_parameter_tree(x, ::AbstractGalerkinBasis) =
    throw(ArgumentError("Unsupported parameter leaf type $(typeof(x)) in Galerkin lifting."))

# ------------------------------------------------------------------------------
# Reconstruction helpers: coefficient tree alpha + basis values psi(t) -> ordinary parameter tree
# ------------------------------------------------------------------------------

# Recursively reconstruct an ordinary parameter tree by applying the Galerkin expansion at each leaf.
reconstruct_parameter_tree(alpha::NamedTuple, psi::AbstractVector) =
    map(v -> reconstruct_parameter_tree(v, psi), alpha)

# Same for Tuple structures.
reconstruct_parameter_tree(alpha::Tuple, psi::AbstractVector) =
    map(v -> reconstruct_parameter_tree(v, psi), alpha)

# Base case: if the parameter leaf is `nothing`, we can just return `nothing` in the reconstructed tree
reconstruct_parameter_tree(::Nothing, ::AbstractVector) = nothing

# Base case: handles array-valued parameter leaves, which are reconstructed via a linear combination of the basis slices weighted by the basis values psi.
@inline function reconstruct_parameter_tree(alpha::AbstractArray, psi::AbstractVector)
    # Check that the number of basis functions matches the leading dimension of the coefficient array.
    length(psi) == size(alpha, 1) || throw(DimensionMismatch(
        "basis length $(length(psi)) does not match lifted parameter size $(size(alpha, 1))."
    ))

    # Scalar leaf case: a base scalar parameter becomes an m-vector of coefficients.
    if ndims(alpha) == 1
        return LinearAlgebra.dot(alpha, psi)
    end

    # Fast path for ordinary dense/strided arrays.
    if alpha isa StridedArray
        m = length(psi)
        flat = reshape(alpha, m, :)
        return reshape(flat' * psi, Base.tail(size(alpha)))
    end

    # Generic fallback for array types without a convenient strided contraction.
    y = psi[1] .* selectdim(alpha, 1, 1)
    @inbounds for j in 2:lastindex(psi)
        y = y .+ psi[j] .* selectdim(alpha, 1, j)
    end
    return y
end

reconstruct_parameter_tree(alpha, ::AbstractVector) =
    throw(ArgumentError("Unsupported coefficient leaf type $(typeof(alpha)) in Galerkin reconstruction."))

# ------------------------------------------------------------------------------
# Layer call
# ------------------------------------------------------------------------------
# The main call method, which constructs the ODE problem and solves it using the reconstructed parameters at each RHS evaluation.
# With x as the initial state, alpha as the coefficient tree of trainable parameters, and st as the initial state tree
function (g::GalerkinNeuralODE)(x, alpha, st)
    # The model is a stateful wrapper around the base model
    model = StatefulLuxLayer{fixed_state_type(g.model)}(g.model, nothing, st)

    # The ODE RHS function, which reconstructs the parameter tree at the current time and evaluates the model to get the state derivative.
    function dudt(u, alpha, t)
        psi = basis_eval(g.basis, t, g.tspan)
        p_t = reconstruct_parameter_tree(alpha, psi)
        return model(u, p_t)
    end

    # IMPORTANT: Unlike current `NeuralODE`, we do not pass `tgrad = basic_tgrad` here,
    # because the RHS depends on time/depth through the reconstructed parameter tree p(t).
    ff = ODEFunction{false}(dudt)
    prob = ODEProblem{false}(ff, x, g.tspan, alpha)

    # Solve the ODE problem using the InterpolatingAdjoint, which will allow gradients to flow through the solver and the time-dependent parameters.
    return (
        solve(
            prob,
            g.args...;
            # sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
            g.kwargs...,
        ),
        model.st,
    )
end
