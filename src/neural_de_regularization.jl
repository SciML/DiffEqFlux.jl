struct ArrayAndTime{A <: AbstractArray, T <: Real}
    array::A
    scalar::T
end

function Lux.apply(l::AbstractLuxLayer, x::ArrayAndTime, ps, st::NamedTuple)
    y, st_ = Lux.apply(l, x.array, ps, st)
    ArrayAndTime(y, x.scalar), st_
end

struct TDChain{L <: NamedTuple} <: AbstractLuxWrapperLayer{:layers}
    layers::L

    TDChain(c::Chain) = TDChain(c.layers)
    TDChain(; kwargs...) = TDChain((; kwargs...))
    TDChain(layers::NamedTuple) = new{typeof(layers)}(layers)
end

(c::TDChain)((x, t), ps, st::NamedTuple) = applytdchain(c.layers, x, t, ps, st)

@generated function applytdchain(
        layers::NamedTuple{fields}, x::T, t, ps, st::NamedTuple{fields}
    ) where {fields, T}
    N = length(fields)
    cat_dim = max(ndims(T) - 1, 1)
    x_symbols = vcat([:x], [gensym("x") for _ in 1:N])
    st_symbols = [gensym("st") for _ in 1:N]
    size_expr = Expr(
        :tuple,
        [i == cat_dim ? 1 : :(size(x, $i)) for i in 1:ndims(T)]...,
    )
    calls = [
        :(
            _t = zero.(similar(x, eltype(x), $size_expr)) .+ convert(eltype(x), t)
        ),
    ]

    for i in 1:N
        field = fields[i]
        push!(
            calls,
            :(
                ($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(
                    layers.$field,
                    cat($(x_symbols[i]), _t; dims = $cat_dim),
                    ps.$field,
                    st.$field,
                )
            ),
        )
    end

    st_out = Expr(:tuple, st_symbols...)
    push!(
        calls,
        :(return ($(x_symbols[N + 1]), _t), NamedTuple{$fields}($st_out)),
    )
    return Expr(:block, calls...)
end

function Lux.apply(l::TDChain, x::ArrayAndTime, ps, st::NamedTuple)
    (y, _), st_ = Lux.apply(l, (x.array, x.scalar), ps, st)
    ArrayAndTime(y, x.scalar), st_
end

@concrete struct RegularizedNeuralODE{R} <: NeuralDELayer
    model <: AbstractLuxLayer
    tspan
    args
    kwargs
end

LuxCore.initialstates(rng::AbstractRNG, n::RegularizedNeuralODE) = (;
    model = LuxCore.initialstates(rng, n.model),
    nfe = -1,
    reg_val = 0.0f0,
    rng = Lux.replicate(rng),
    training = Val(true),
)

mutable struct ReservoirState{R, T, U}
    rng::R
    count::Int
    pending_t::Union{Nothing, T}
    pending_u::Union{Nothing, U}
    treg::Union{Nothing, T}
    ureg::Union{Nothing, U}
end

function ReservoirState(rng, t::T, u::U, save_start::Bool) where {T, U}
    pending_t = save_start ? t : nothing
    pending_u = if save_start
        u isa Number ? u : copyto!(similar(u), u)
    else
        nothing
    end
    ReservoirState{typeof(rng), T, U}(rng, 0, pending_t, pending_u, nothing, nothing)
end

struct ReservoirCallback{S}
    state::S
end

function update_reservoir!(state::ReservoirState, t, u)
    if state.pending_t !== nothing
        state.count += 1
        if state.count == 1 || rand(state.rng, 1:state.count) == 1
            state.treg = state.pending_t
            pending_u = state.pending_u
            state.ureg = if pending_u isa Number
                pending_u
            elseif state.ureg === nothing
                copyto!(similar(pending_u), pending_u)
            else
                copyto!(state.ureg, pending_u)
            end
        end
    end

    state.pending_t = t
    state.pending_u = if u isa Number
        u
    elseif state.pending_u === nothing
        copyto!(similar(u), u)
    else
        copyto!(state.pending_u, u)
    end
    return nothing
end

function (affect::ReservoirCallback)(integrator)
    update_reservoir!(affect.state, integrator.t, integrator.u)
    SciMLBase.u_modified!(integrator, false)
    nothing
end

function reservoir_callback(state::ReservoirState)
    SciMLBase.DiscreteCallback(
        (u, t, integrator) -> integrator.iter > 0,
        ReservoirCallback(state);
        save_positions = (false, false),
    )
end

function SciMLSensitivity._track_callback(
        cb::SciMLBase.DiscreteCallback{C, A}, t, u, p, sensealg
    ) where {C, A <: ReservoirCallback}
    cb
end

function SciMLSensitivity._setup_reverse_callbacks(
        cb::SciMLBase.DiscreteCallback{C, A},
        affect::A, sensealg, dgdu,
        dgdp,
        loss_ref, terminated
    ) where {C, A <: ReservoirCallback}
    SciMLBase.DiscreteCallback(
        (u, t, integrator) -> false,
        integrator -> nothing;
        save_positions = (false, false),
    )
end

CRC.@non_differentiable update_reservoir!(::Any...)
CRC.@non_differentiable reservoir_callback(::Any...)

function solve_node(n::RegularizedNeuralODE, x, p, st, kwargs)
    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st.model)

    dudt(u, p, t) = model(ArrayAndTime(u, t), p).array
    function dudt!(du, u, p, t)
        copyto!(du, dudt(u, p, t))
        return nothing
    end

    prob = ODEProblem{true}(ODEFunction{true}(dudt!), x, n.tspan, p)
    local_prob = ODEProblem{false}(ODEFunction{false}(dudt; tgrad = basic_tgrad), x, n.tspan, p)
    sol = solve(
        prob, n.args...;
        sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), kwargs...
    )
    sol, model.st, local_prob
end

function local_regularization_step end

function vanilla_node(n::RegularizedNeuralODE, x, p, st)
    solve_kwargs = haskey(n.kwargs, :saveat) ?
        n.kwargs :
        merge(n.kwargs, (; saveat = [last(n.tspan)]))
    sol, model_st, _ = solve_node(n, x, p, st, solve_kwargs)
    sol, (; model = model_st, nfe = sol.destats.nf, reg_val = 0.0f0, st.rng, st.training)
end

function biased_node(n::RegularizedNeuralODE, x, p, st)
    rng = Lux.replicate(st.rng)
    kws = n.kwargs
    save_start = get(kws, :save_start, true)
    reservoir = CRC.@ignore_derivatives ReservoirState(rng, first(n.tspan), x, save_start)
    solve_kwargs = CRC.@ignore_derivatives begin
        callback = reservoir_callback(reservoir)
        solve_kwargs = haskey(kws, :callback) ?
            merge(kws, (; callback = SciMLBase.CallbackSet(kws[:callback], callback))) :
            merge(kws, (; callback))
        if !haskey(kws, :saveat) && !haskey(kws, :save_everystep)
            solve_kwargs = merge(solve_kwargs, (; save_everystep = false, save_end = true))
        end
        solve_kwargs
    end

    sol, model_st, prob = solve_node(n, x, p, st, solve_kwargs)
    treg, ureg = CRC.@ignore_derivatives (reservoir.treg, reservoir.ureg)
    (treg === nothing || ureg === nothing) &&
        error("Biased regularization requires at least one accepted-step candidate.")

    integrator = CRC.@ignore_derivatives begin
        local_prob = remake(prob; u0 = ureg, tspan = (treg, last(prob.tspan)))
        SciMLBase.init(local_prob, n.args...; kws...)
    end
    reg_val, local_nf = local_regularization_step(integrator, p)
    nfe = sol.destats.nf + local_nf

    sol, (; model = model_st, nfe, reg_val, rng, st.training)
end

function unbiased_node(n::RegularizedNeuralODE, x, p, st)
    rng = Lux.replicate(st.rng)
    t0, t1 = n.tspan
    treg = CRC.@ignore_derivatives rand(rng, typeof(t1 - t0)) * (t1 - t0) + t0
    solve_kwargs, needs_correction = CRC.@ignore_derivatives begin
        if haskey(n.kwargs, :saveat)
            saveat = n.kwargs[:saveat] isa Number ? [n.kwargs[:saveat]] : collect(n.kwargs[:saveat])
            idx = findfirst(isequal(treg), saveat)
            idx === nothing && push!(saveat, treg)
            sort!(saveat)
            merge(n.kwargs, (; saveat)), idx === nothing
        else
            merge(n.kwargs, (; saveat = sort!([treg, last(n.tspan)]))), false
        end
    end
    sol, model_st, prob = solve_node(n, x, p, st, solve_kwargs)
    idx = CRC.@ignore_derivatives begin
        idx = findfirst(isequal(treg), sol.t)
        idx === nothing &&
            error("Failed to recover unbiased regularization state from the saved solution grid.")
        idx
    end
    ureg = sol.u[idx]
    needs_correction && (
        sol = CRC.@ignore_derivatives SciMLBase.solution_slice(
            sol, filter(i -> i != idx, eachindex(sol.t))
        )
    )
    integrator = CRC.@ignore_derivatives begin
        local_prob = remake(prob; u0 = ureg, tspan = (treg, last(prob.tspan)))
        SciMLBase.init(local_prob, n.args...; n.kwargs...)
    end
    reg_val, local_nf = local_regularization_step(integrator, p)
    nfe = sol.destats.nf + local_nf

    sol, (; model = model_st, nfe, reg_val, rng, st.training)
end

(n::RegularizedNeuralODE)(x, p, st) = n(x, p, st, st.training)

(n::RegularizedNeuralODE{:none})(x, p, st, ::Val) = vanilla_node(
    n, x, p, st
)

(n::RegularizedNeuralODE{:unbiased})(x, p, st, ::Val{false}) = vanilla_node(
    n, x, p, st
)

(n::RegularizedNeuralODE{:unbiased})(x, p, st, ::Val{true}) = unbiased_node(
    n, x, p, st
)

(n::RegularizedNeuralODE{:biased})(x, p, st, ::Val{false}) = vanilla_node(
    n, x, p, st
)

(n::RegularizedNeuralODE{:biased})(x, p, st, ::Val{true}) = biased_node(
    n, x, p, st
)

