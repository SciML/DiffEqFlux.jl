@concrete struct RegularizedNeuralODE{R} <: NeuralDELayer
    model <: AbstractLuxLayer
    tspan
    args
    kwargs
end

function regularized_neuralode(model, tspan, regularize, args, kwargs)
    regularize in (:none, :unbiased, :biased) ||
        throw(ArgumentError(
            "regularize must be one of (:none, :unbiased, :biased), " *
            "or `nothing` to use the ordinary NeuralODE."
        ))
    return RegularizedNeuralODE{regularize}(model, tspan, args, kwargs)
end

LuxCore.initialstates(rng::AbstractRNG, n::RegularizedNeuralODE) = (;
    model = LuxCore.initialstates(rng, n.model),
    nfe = -1,
    reg_val = 0.0f0,
    rng = Lux.replicate(rng),
    training = Val(true),
)

function solve_neuralode(n::RegularizedNeuralODE, x, p, st, kwargs)
    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st.model)
    dudt(u, p, t) = model((u, t), p)
    function dudt!(du, u, p, t)
        copyto!(du, dudt(u, p, t))
        return nothing
    end

    prob = ODEProblem{true}(ODEFunction{true}(dudt!), x, n.tspan, p)
    sol = solve(
        prob, n.args...;
        sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), kwargs...
    )
    sol, model.st
end

function error_estimate_regularization end

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
    pending_u = save_start ? copyto!(similar(u), u) : nothing
    return ReservoirState{typeof(rng), T, U}(rng, 0, pending_t, pending_u, nothing, nothing)
end

function (state::ReservoirState)(integrator)
    if state.pending_t !== nothing
        state.count += 1
        if state.count == 1 || rand(state.rng, 1:(state.count)) == 1
            state.treg = state.pending_t
            pending_u = state.pending_u
            state.ureg === nothing && (state.ureg = similar(pending_u))
            copyto!(state.ureg, pending_u)
        end
    end

    state.pending_t = integrator.t
    state.pending_u === nothing && (state.pending_u = similar(integrator.u))
    copyto!(state.pending_u, integrator.u)
    SciMLBase.u_modified!(integrator, false)
    return nothing
end

function SciMLSensitivity._track_callback(
        cb::SciMLBase.DiscreteCallback{C, A}, t, u, p, sensealg
    ) where {C, A <: ReservoirState}
    return cb
end

function SciMLSensitivity._setup_reverse_callbacks(
        cb::SciMLBase.DiscreteCallback{C, A},
        affect::A, sensealg, dgdu,
        dgdp,
        loss_ref, terminated
    ) where {C, A <: ReservoirState}
    return SciMLBase.DiscreteCallback(
        (u, t, integrator) -> false,
        integrator -> nothing;
        save_positions = (false, false),
    )
end

function local_regularization_from_state(n, x, p, st, kws, treg, ureg)
    model, integrator = CRC.@ignore_derivatives begin
        model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st.model)
        dudt(u, p, t) = model((u, t), p)
        prob = ODEProblem{false}(
            ODEFunction{false}(dudt; tgrad = basic_tgrad), x, n.tspan, p
        )
        local_prob = remake(prob; u0 = ureg, tspan = (treg, last(prob.tspan)))
        model, SciMLBase.init(local_prob, n.args...; kws...)
    end
    return error_estimate_regularization(integrator, model, p)
end

function CRC.rrule(
        config::CRC.RuleConfig{>:CRC.HasReverseMode},
        ::typeof(local_regularization_from_state),
        n, x, p, st, kws, treg, ureg
    )
    model, integrator = CRC.@ignore_derivatives begin
        model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st.model)
        dudt(u, p, t) = model((u, t), p)
        prob = ODEProblem{false}(
            ODEFunction{false}(dudt; tgrad = basic_tgrad), x, n.tspan, p
        )
        local_prob = remake(prob; u0 = ureg, tspan = (treg, last(prob.tspan)))
        model, SciMLBase.init(local_prob, n.args...; kws...)
    end
    y, local_pullback = CRC.rrule(config, error_estimate_regularization, integrator, model, p)

    function local_regularization_from_state_pullback(Delta)
        dlocal = local_pullback(Delta)
        dp = CRC.unthunk(dlocal[4])
        return (
            CRC.NoTangent(),
            CRC.NoTangent(),
            CRC.NoTangent(),
            dp,
            CRC.NoTangent(),
            CRC.NoTangent(),
            CRC.NoTangent(),
            CRC.NoTangent(),
        )
    end
    return y, local_regularization_from_state_pullback
end

function solve_without_regularization(n::RegularizedNeuralODE, x, p, st)
    kws = n.kwargs
    solve_kwargs = haskey(kws, :saveat) ?
        kws :
        merge(kws, (; saveat = [last(n.tspan)]))
    sol, model_st = solve_neuralode(n, x, p, st, solve_kwargs)
    sol, (; model = model_st, nfe = sol.destats.nf, reg_val = 0.0f0, st.rng, st.training)
end

function (n::RegularizedNeuralODE{:none})(x, p, st)
    return solve_without_regularization(n, x, p, st)
end

function solve_biased_regularization(n::RegularizedNeuralODE, x, p, st)
    rng = Lux.replicate(st.rng)
    kws = n.kwargs
    reservoir = CRC.@ignore_derivatives ReservoirState(
        rng, first(n.tspan), x, get(kws, :save_start, true)
    )
    solve_kwargs = CRC.@ignore_derivatives begin
        callback = SciMLBase.DiscreteCallback(
            (u, t, integrator) -> integrator.iter > 0,
            reservoir;
            save_positions = (false, false),
        )
        solve_kwargs = if haskey(kws, :callback) && kws[:callback] !== nothing
            merge(kws, (; callback = SciMLBase.CallbackSet(kws[:callback], callback)))
        else
            merge(kws, (; callback))
        end
        if !haskey(kws, :saveat) &&
           !haskey(kws, :save_everystep) &&
           !(haskey(kws, :dense) && kws[:dense] === true)
            solve_kwargs = merge(solve_kwargs, (; save_everystep = false, save_end = true))
        end
        solve_kwargs
    end
    sol, model_st = solve_neuralode(n, x, p, st, solve_kwargs)

    treg, ureg = CRC.@ignore_derivatives begin
        (reservoir.treg === nothing || reservoir.ureg === nothing) &&
            error("Biased regularization requires at least one accepted-step candidate.")
        reservoir.treg, reservoir.ureg
    end
    reg_val, local_nf = local_regularization_from_state(n, x, p, st, kws, treg, ureg)
    nfe = sol.destats.nf + local_nf

    sol, (; model = model_st, nfe, reg_val, rng, st.training)
end

function solve_unbiased_regularization(n::RegularizedNeuralODE, x, p, st)
    rng = Lux.replicate(st.rng)
    t0, t1 = n.tspan
    kws = n.kwargs
    treg = CRC.@ignore_derivatives rand(rng, typeof(t1 - t0)) * (t1 - t0) + t0
    solve_kwargs = CRC.@ignore_derivatives begin
        saveat = if haskey(kws, :saveat) && kws[:saveat] !== nothing
            saveat = kws[:saveat] isa Number ? [kws[:saveat]] : collect(kws[:saveat])
            any(isequal(treg), saveat) || push!(saveat, treg)
            saveat
        else
            [treg, last(n.tspan)]
        end
        merge(kws, (; saveat = sort!(saveat)))
    end
    sol, model_st = solve_neuralode(n, x, p, st, solve_kwargs)
    idx = CRC.@ignore_derivatives findfirst(isequal(treg), sol.t)
    idx === nothing &&
        error("Failed to recover unbiased regularization state from the saved solution grid.")
    ureg = CRC.@ignore_derivatives copy(sol.u[idx])
    reg_val, local_nf = local_regularization_from_state(n, x, p, st, kws, treg, ureg)
    nfe = sol.destats.nf + local_nf

    sol, (; model = model_st, nfe, reg_val, rng, st.training)
end

function (n::RegularizedNeuralODE{:unbiased})(x, p, st)
    if st.training === Val(true)
        return solve_unbiased_regularization(n, x, p, st)
    else
        return solve_without_regularization(n, x, p, st)
    end
end

function (n::RegularizedNeuralODE{:biased})(x, p, st)
    if st.training === Val(true)
        return solve_biased_regularization(n, x, p, st)
    else
        return solve_without_regularization(n, x, p, st)
    end
end
