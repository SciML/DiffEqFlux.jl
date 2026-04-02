struct ArrayAndTime{A, T <: Real}
    array::A
    scalar::T
end

function Lux.apply(l::AbstractLuxLayer, x::ArrayAndTime, ps, st::NamedTuple)
    y, st_ = Lux.apply(l, x.array, ps, st)
    return ArrayAndTime(y, x.scalar), st_
end

struct TDChain{L <: NamedTuple} <: AbstractLuxWrapperLayer{:layers}
    layers::L

    TDChain(c::Chain) = new{typeof(c.layers)}(c.layers)
    TDChain(; kwargs...) = new{typeof((; kwargs...))}((; kwargs...))
    TDChain(layers::NamedTuple) = new{typeof(layers)}(layers)
end

function _time_channel_like(x::AbstractArray, t, dims)
    return fill!(similar(x, eltype(x), dims), convert(eltype(x), t))
end

CRC.@non_differentiable _time_channel_like(::Any...)

function (c::TDChain)((x, t), ps, st::NamedTuple)
    return applytdchain(c.layers, x, t, ps, st)
end

@generated function applytdchain(layers::NamedTuple{fields}, x::T, t, ps, st) where {fields, T}
    N = length(fields)
    cat_dim = max(ndims(T) - 1, 1)
    x_symbols = [:x; [gensym(:x) for _ in 1:N]]
    st_symbols = [gensym(:st) for _ in 1:N]
    dims = Expr(:tuple, [i == cat_dim ? 1 : :(size(x, $i)) for i in 1:(ndims(T))]...)
    block = Any[:(_t = _time_channel_like(x, t, $dims))]

    for i in 1:N
        field = fields[i]
        push!(
            block,
            :(
                ($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(
                    layers.$field,
                    cat($(x_symbols[i]), _t; dims = Val($cat_dim)),
                    ps.$field,
                    st.$field,
                )
            ),
        )
    end

    st_out = Expr(:tuple, st_symbols...)
    push!(block, :(return ($(x_symbols[N + 1]), _t), NamedTuple{$fields}($st_out)))
    return Expr(:block, block...)
end

function Lux.apply(l::TDChain, x::ArrayAndTime, ps, st::NamedTuple)
    (y, _), st_ = Lux.apply(l, (x.array, x.scalar), ps, st)
    return ArrayAndTime(y, x.scalar), st_
end

@concrete struct RegularizedNeuralODE <: NeuralDELayer
    model <: AbstractLuxLayer
    tspan
    args
    regularize::Symbol
    kwargs
end

    function __construct_neural_ode(model, tspan, args, regularize, kwargs)
        regularize === nothing && return NeuralODE(model, tspan, args, kwargs)
        if regularize isa Bool
        regularize = regularize ? :unbiased : :none
        end
        regularize in (:none, :unbiased, :biased) ||
            throw(ArgumentError("`regularize` must be one of `:none`, `:unbiased`, or `:biased`."))
        return RegularizedNeuralODE(model, tspan, args, regularize, values(kwargs))
    end

LuxCore.initialstates(rng::AbstractRNG, n::RegularizedNeuralODE) = (;
    model = LuxCore.initialstates(rng, n.model),
    nfe = -1,
    reg_val = 0.0f0,
    rng = Lux.replicate(rng),
    training = Val(true),
)

function __ode_nfe(sol)
    if hasproperty(sol, :destats) && hasproperty(sol.destats, :nf)
        return sol.destats.nf
    elseif hasproperty(sol, :stats) && hasproperty(sol.stats, :nf)
        return sol.stats.nf
    else
        return -1
    end
end

CRC.@non_differentiable __ode_nfe(::Any...)

function __regularized_solve(
        n::RegularizedNeuralODE, x, p, st; kwargs = n.kwargs
    )
    model_state = hasproperty(st, :model) ? st.model : st
    model = StatefulLuxLayer{fixed_state_type(n.model)}(
        n.model, nothing, model_state
    )

    dudt(u, p, t) = model(ArrayAndTime(u, t), p).array
    ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)
    sol = solve(
        prob, n.args...;
        sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), kwargs...
    )
    return sol, model.st, prob
end

function local_regularization_step(integrator, cache, p)
    throw(
        ArgumentError(
            "Local regularization is not implemented for solver $(typeof(integrator.alg)) " *
            "with cache $(typeof(cache))."
        ),
    )
end

function __vanilla_regularized_neural_ode(n::RegularizedNeuralODE, x, p, st)
    solve_kwargs = let kws = n.kwargs
        haskey(kws, :saveat) ? kws : merge(kws, (; saveat = [last(n.tspan)]))
    end
    sol, model_st, _ = __regularized_solve(
        n, x, p, st; kwargs = solve_kwargs
    )
    nfe = __ode_nfe(sol)
    rng = hasproperty(st, :rng) ? st.rng : Random.default_rng()
    training = hasproperty(st, :training) ? st.training : Val(true)
    return sol, (; model = model_st, nfe, reg_val = 0.0f0, rng, training)
end

function __biased_regularized_neural_ode(n::RegularizedNeuralODE, x, p, st)
    isempty(n.args) && throw(
        ArgumentError(
            "Local regularization for `NeuralODE` requires an explicit solver algorithm, " *
            "for example `Tsit5()`."
        ),
    )

    solve_kwargs = let kws = n.kwargs
        haskey(kws, :saveat) ? kws : merge(kws, (; saveat = []))
    end
    sol, model_st, prob = __regularized_solve(n, x, p, st; kwargs = solve_kwargs)
    rng = Lux.replicate(hasproperty(st, :rng) ? st.rng : Random.default_rng())
    training = hasproperty(st, :training) ? st.training : Val(true)
    idx = CRC.@ignore_derivatives begin
        isempty(sol.t) && error("Biased regularization requires at least one saved solution state.")
        length(sol.t) == 1 ? firstindex(sol.t) :
        rand(rng, firstindex(sol.t):(lastindex(sol.t) - 1))
    end
    treg = sol.t[idx]
    ureg = sol.u[idx]
    integrator = CRC.@ignore_derivatives begin
        local_prob = remake(prob; u0 = ureg, tspan = (treg, last(prob.tspan)))
        SciMLBase.init(local_prob, n.args...; solve_kwargs...)
    end
    reg_val, local_nf = local_regularization_step(integrator, integrator.cache, p)
    nfe = max(__ode_nfe(sol), 0) + local_nf

    return sol, (; model = model_st, nfe, reg_val, rng, training)
end

function __unbiased_regularized_neural_ode(n::RegularizedNeuralODE, x, p, st)
    isempty(n.args) && throw(
        ArgumentError(
            "Local regularization for `NeuralODE` requires an explicit solver algorithm, " *
            "for example `Tsit5()`."
        ),
    )

    rng = Lux.replicate(hasproperty(st, :rng) ? st.rng : Random.default_rng())
    training = hasproperty(st, :training) ? st.training : Val(true)
    t0, t1 = n.tspan
    treg = CRC.@ignore_derivatives rand(rng, typeof(t1 - t0)) * (t1 - t0) + t0
    solve_kwargs, needs_correction = CRC.@ignore_derivatives begin
        kws = n.kwargs
        if haskey(kws, :saveat)
            saveat = collect(kws[:saveat])
            needs_correction = findfirst(isequal(treg), saveat) === nothing
            needs_correction && push!(saveat, treg)
            sort!(saveat)
            merge(kws, (; saveat)), needs_correction
        else
            saveat = sort!([treg, last(n.tspan)])
            merge(kws, (; saveat)), false
        end
    end
    sol, model_st, prob = __regularized_solve(n, x, p, st; kwargs = solve_kwargs)
    t_idx = CRC.@ignore_derivatives begin
        idx = findfirst(isequal(treg), sol.t)
        idx === nothing &&
            error("Failed to recover unbiased regularization point from saved solution grid.")
        idx
    end
    ureg = sol.u[t_idx]
    integrator = CRC.@ignore_derivatives begin
        local_prob = remake(prob; u0 = ureg, tspan = (treg, last(prob.tspan)))
        SciMLBase.init(local_prob, n.args...; solve_kwargs...)
    end
    reg_val, local_nf = local_regularization_step(integrator, integrator.cache, p)
    nfe = max(__ode_nfe(sol), 0) + local_nf
    sol_out = if needs_correction
        keep = CRC.@ignore_derivatives filter(i -> i != t_idx, eachindex(sol.t))
        CRC.@ignore_derivatives SciMLBase.solution_slice(sol, keep)
    else
        sol
    end

    return sol_out, (; model = model_st, nfe, reg_val, rng, training)
end

function (n::RegularizedNeuralODE)(x, p, st)
    training = hasproperty(st, :training) ? st.training : Val(true)
    return n(x, p, st, training)
end

(n::RegularizedNeuralODE)(x, p, st, ::Val{false}) = __vanilla_regularized_neural_ode(n, x, p, st)

function (n::RegularizedNeuralODE)(x, p, st, ::Val{true})
    return if n.regularize === :none
        __vanilla_regularized_neural_ode(n, x, p, st)
    elseif n.regularize === :unbiased
        __unbiased_regularized_neural_ode(n, x, p, st)
    else
        __biased_regularized_neural_ode(n, x, p, st)
    end
end
