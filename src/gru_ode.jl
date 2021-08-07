abstract type AbstractGRUODELayer <: Function end
Flux.trainable(m::AbstractGRUODELayer) = (m.p,)

struct GRUODELayer{P,RE,T,A,K} <: AbstractGRUODELayer
    model::Flux.GRUCell
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K

    function GRUODELayer(model,tspan,args...;p = nothing,kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,tspan,args,kwargs)
    end

    function GRUODELayer(model::FastChain,tspan,args...;p = initial_params(model),kwargs...)
        re = nothing
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,tspan,args,kwargs)
    end
end

function (n::GRUODELayer)(x,p=n.p)
    function dudt_(u,p,t)
        gx, gu, r, z = Flux._gru_output(n.model.Wi, n.model.Wh, x, u)
        g = tanh.(Flux.gate(gx, o, 3) .+ r .* Flux.gate(gh, o, 3) .+ Flux.gate(b, o, 3))
        return (1 .- z) .* (g .- u)
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;sense=sense,n.kwargs...)
end