abstract type NeuralDELayer <: Function end

"""
Constructs a neural ODE with the gradients computed using the adjoint
method[1]. At a high level this corresponds to solving the forward
differential equation, using a second differential equation that propagates
the derivatives of the loss  backwards in time.
This first solves the continuous time problem, and then discretizes following
the rules specified by the numerical ODE solver.
On the other hand, the 'neural_ode_rd' first disretizes the solution and then
computes the adjoint using automatic differentiation.

Ref
[1]L. S. Pontryagin, Mathematical Theory of Optimal Processes. CRC Press, 1987.

Arguments
≡≡≡≡≡≡≡≡
model::Chain defines the ̇x
x<:AbstractArray initial value x(t₀)
args arguments passed to ODESolve
kwargs key word arguments passed to ODESolve; accepts an additional key
    :callback_adj in addition to :callback. The Callback :callback_adj
    passes a separate callback to the adjoint solver.

"""
struct NeuralODE{M,P,RE,T,S,A,K} <: NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    solver::S
    args::A
    kwargs::K

    function NeuralODE(model,tspan,solver=nothing,args...;kwargs...)
        p,re = Flux.destructure(model)
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(solver),typeof(args),typeof(kwargs)}(
            model,p,re,tspan,solver,args,kwargs)
    end

    function NeuralODE(model::FastChain,tspan,solver=nothing,args...;kwargs...)
        p = initial_params(model)
        re = nothing
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(solver),typeof(args),typeof(kwargs)}(
            model,p,re,tspan,solver,args,kwargs)
    end
end

Flux.@functor NeuralODE

function (n::NeuralODE)(x,p=n.p)
    dudt_(u,p,t) = n.re(p)(u)
    prob = ODEProblem{false}(dudt_,x,n.tspan,p)
    concrete_solve(prob,n.solver,x,p,n.args...;n.kwargs...)
end

function (n::NeuralODE{M})(x,p=n.p) where {M<:FastChain}
    dudt_(u,p,t) = n.model(u,p)
    prob = ODEProblem{false}(dudt_,x,n.tspan,p)
    concrete_solve(prob,n.solver,x,p,n.args...;n.kwargs...)
end

struct NeuralDSDE{M,P,RE,M2,RE2,T,S,A,K} <: NeuralDELayer
    p::P
    len::Int
    model1::M
    re1::RE
    model2::M2
    re2::RE2
    tspan::T
    solver::S
    args::A
    kwargs::K
    function NeuralDSDE(model1,model2,tspan,solver=nothing,args...;kwargs...)
        p1,re1 = Flux.destructure(model1)
        p2,re2 = Flux.destructure(model2)
        p = [p1;p2]
        new{typeof(model1),typeof(p),typeof(re1),typeof(model2),typeof(re2),
            typeof(tspan),typeof(solver),typeof(args),typeof(kwargs)}(p,
            length(p1),model1,re1,model2,re2,tspan,solver,args,kwargs)
    end

    function NeuralDSDE(model1::FastChain,model2::FastChain,tspan,solver=nothing,args...;kwargs...)
        p1 = initial_params(model1)
        p2 = initial_params(model2)
        re1 = nothing
        re2 = nothing
        p = [p1;p2]
        new{typeof(model1),typeof(p),typeof(re1),typeof(model2),typeof(re2),
            typeof(tspan),typeof(solver),typeof(args),typeof(kwargs)}(p,
            length(p1),model1,re1,model2,re2,tspan,solver,args,kwargs)
    end
end

Flux.@functor NeuralDSDE

function (n::NeuralDSDE)(x,p=n.p)
    dudt_(u,p,t) = n.re1(p[1:n.len])(u)
    g(u,p,t) = n.re2(p[(n.len+1):end])(u)
    prob = SDEProblem{false}(dudt_,g,x,n.tspan,p)
    concrete_solve(prob,n.solver,x,p,n.args...;sensealg=TrackerAdjoint(),n.kwargs...)
end

function (n::NeuralDSDE{M})(x,p=n.p) where {M<:FastChain}
    dudt_(u,p,t) = n.model1(u,p)
    g(u,p,t) = n.model2(u,p)
    prob = SDEProblem{false}(dudt_,g,x,n.tspan,p)
    concrete_solve(prob,n.solver,x,p,n.args...;sensealg=TrackerAdjoint(),n.kwargs...)
end

struct NeuralSDE{P,M,RE,M2,RE2,T,S,A,K} <: NeuralDELayer
    p::P
    len::Int
    model1::M
    re1::RE
    model2::M2
    re2::RE2
    tspan::T
    nbrown::Int
    solver::S
    args::A
    kwargs::K
end

function NeuralSDE(model1,model2,tspan,nbrown,solver=nothing,args...;kwargs...)
    p1,re1 = Flux.destructure(model1)
    p2,re2 = Flux.destructure(model2)
    p = [p1;p2]
    NeuralSDE(p,length(p1),model1,re1,model2,re2,tspan,nbrown,solver,args,kwargs)
end

Flux.@functor NeuralSDE

function (n::NeuralSDE)(x,p=n.p)
    dudt_(u,p,t) = n.re1(p[1:n.len])(u)
    g(u,p,t) = n.re2(p[(n.len+1):end])(u)
    prob = SDEProblem{false}(dudt_,g,x,n.tspan,p,noise_rate_prototype=zeros(Float32,length(x),n.nbrown))
    concrete_solve(prob,n.solver,x,p,n.args...;sensealg=TrackerAdjoint(),n.kwargs...)
end

struct NeuralCDDE{P,M,RE,H,L,T,S,A,K} <: NeuralDELayer
    p::P
    model::M
    re::RE
    hist::H
    lags::L
    tspan::T
    solver::S
    args::A
    kwargs::K
end

function NeuralCDDE(model,tspan,hist,lags,solver=nothing,args...;kwargs...)
    p,re = Flux.destructure(model)
    NeuralCDDE(p,model,re,hist,lags,tspan,solver,args,kwargs)
end

Flux.@functor NeuralCDDE

function (n::NeuralCDDE)(x,p=n.p)
    function dudt_(u,h,p,t)
        _u = vcat(u,(h(p,t-lag) for lag in n.lags)...)
        n.re(p)(_u)
    end
    prob = DDEProblem{false}(dudt_,x,n.hist,n.tspan,p,constant_lags = n.lags)
    concrete_solve(prob,n.solver,x,p,n.args...;sensealg=TrackerAdjoint(),n.kwargs...)
end
