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
        solve(prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), n.kwargs...),
        model.st)
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
        n.diffusion, nothing, st.diffusion)

    dudt(u, p, t) = drift(u, p.drift)
    g(u, p, t) = diffusion(u, p.diffusion)

    ff = SDEFunction{false}(dudt, g; tgrad = basic_tgrad)
    prob = SDEProblem{false}(ff, g, x, n.tspan, p)
    return (solve(prob, n.args...; u0 = x, p, sensealg = TrackerAdjoint(), n.kwargs...),
        (; drift = drift.st, diffusion = diffusion.st))
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
        n.diffusion, p.diffusion, st.diffusion)

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
        ff, x, (p, t) -> n.hist(x, p, t), n.tspan, ps; constant_lags = n.lags)

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
        model, constraints_model, tspan, args...; differential_vars = nothing, kwargs...)
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
        solve(prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), n.kwargs...),
        model.st)
end

"""
    ODERNN(model, cell, args...; ordering=BatchLastIndex(), return_sequence=false, kwargs...)

Constructs an ODE-RNN layer that combines a Neural ODE for continuous hidden state dynamics
with a recurrent neural network cell for processing sequential observations. This implements
a variant of the ODE-RNN/ODE-LSTM architecture from [1, 2].

The layer processes sequential data by:
1. Initializing a hidden state from the first input using the RNN cell
2. Solving a Neural ODE from the first to last time point to get continuous hidden dynamics
3. At each time step, using the ODE-evolved hidden state with the RNN cell to process inputs

Arguments:

  - `model`: A `Flux.Chain` or `Lux.AbstractLuxLayer` neural network that defines the ODE
    dynamics dh/dt = model(h). Should map hidden_size -> hidden_size.
  - `cell`: A `Lux.AbstractRecurrentCell` (e.g., `LSTMCell`, `GRUCell`, `RNNCell`) that
    processes inputs at each time step.
  - `args`: Additional positional arguments passed to the ODE solver.
  - `ordering`: The time series data batch ordering. Defaults to `BatchLastIndex()`.
  - `return_sequence`: If `true`, returns outputs at all time steps. If `false`, returns
    only the final output. Defaults to `false`.
  - `sensealg`: The choice of differentiation algorithm used in the backpropagation.
    Defaults to `InterpolatingAdjoint(; autojacvec = ZygoteVJP())`.
  - `kwargs`: Additional keyword arguments splatted to the ODE solver.

The forward pass takes a tuple `(x, ts)` where:
  - `x`: Input array of shape `(input_dim, seq_len, batch)` with `BatchLastIndex()` ordering
  - `ts`: Vector of time points corresponding to each element in the sequence

Example:

```julia
using DiffEqFlux, Lux, Random

rng = Random.default_rng()
input_dim, hidden_dim, seq_len, batch_size = 2, 4, 10, 3

# ODE dynamics model (hidden_dim -> hidden_dim)
ode_model = Chain(Dense(hidden_dim => hidden_dim, tanh))

# RNN cell (processes input_dim -> hidden_dim)
cell = LSTMCell(input_dim => hidden_dim)

# Create ODE-RNN layer
odernn = ODERNN(ode_model, cell; return_sequence=true)

# Setup
ps, st = Lux.setup(rng, odernn)

# Input data and time points
x = randn(Float32, input_dim, seq_len, batch_size)
ts = collect(Float32, range(0, 1, length=seq_len))

# Forward pass
outputs, st = odernn((x, ts), ps, st)
```

References:

[1] Rubanova, Yulia, Ricky TQ Chen, and David Duvenaud. "Latent ordinary differential
    equations for irregularly-sampled time series." Advances in neural information
    processing systems 32 (2019).

[2] Lechner, Mathias, and Ramin Hasani. "Learning long-term dependencies in irregularly-sampled
    time series." arXiv preprint arXiv:2006.04418 (2020).
"""
@concrete struct ODERNN <: AbstractLuxContainerLayer{(:model, :cell)}
    model <: AbstractLuxLayer
    cell <: AbstractLuxLayer
    return_sequence
    ordering
    args
    kwargs
end

function ODERNN(model, cell, args...;
        ordering = Lux.BatchLastIndex(),
        return_sequence::Bool = false,
        kwargs...)
    !(model isa AbstractLuxLayer) && (model = FromFluxAdaptor()(model))
    !(cell isa AbstractLuxLayer) && (cell = FromFluxAdaptor()(cell))
    return ODERNN(model, cell, static(return_sequence), ordering, args, kwargs)
end

function (odernn::ODERNN)((x, ts)::Tuple{<:AbstractArray, <:AbstractVector}, ps, st)
    xs = Lux.safe_eachslice(x, odernn.ordering)

    # Initialize with first input to get initial hidden state
    x0 = first(xs)
    (_, init_carry), st_cell = odernn.cell(x0, ps.cell, st.cell)
    h0 = first(init_carry)

    # Solve Neural ODE once from first to last time point
    model = StatefulLuxLayer{fixed_state_type(odernn.model)}(odernn.model, nothing, st.model)
    dudt(u, p, t) = model(u, p)
    ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
    tspan = (first(ts), last(ts))
    prob = ODEProblem{false}(ff, h0, tspan, ps.model)
    sol = solve(prob, odernn.args...;
        saveat = ts,
        sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
        odernn.kwargs...)

    # Process sequence with RNN cell, using ODE solutions as hidden states
    carry = init_carry
    outputs = __odernn_loop(odernn, xs, sol, carry, ps, st_cell)

    new_st = (; model = model.st, cell = st_cell)

    if known(odernn.return_sequence)
        return outputs, new_st
    else
        return last(outputs), new_st
    end
end

function __odernn_loop(odernn, xs, sol, carry, ps, st_cell)
    # Use foldl to accumulate outputs in a differentiable way
    init = (carry, st_cell, ())
    result = foldl(enumerate(xs); init = init) do (carry, st_cell, outputs), (i, x_i)
        # Get ODE solution at this time point and update carry
        ode_h = sol.u[i]
        carry = (ode_h, Base.tail(carry)...)

        # RNN cell processes input with ODE-evolved carry
        (out, carry), st_cell = odernn.cell((x_i, carry), ps.cell, st_cell)
        return (carry, st_cell, (outputs..., out))
    end
    return result[3]
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
        similar(x, size(x)[1:(ndims(x) - 2)]..., augment_dim, size(x, ndims(x))), 0)
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
