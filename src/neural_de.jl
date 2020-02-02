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
function neural_ode(model,x,tspan,args...;kwargs...)
    error("neural_ode has been deprecated with the change to Zygote. Please see the documentation on the new NeuralODE layer.")
end

"""
Constructs a neural ODE with the gradients computed using  reverse-mode
automatic differentiation. This is equivalent to discretizing then optimizing
the differential equation, cf neural_ode for a comparison with the adjoint method.
"""
function neural_ode_rd(model,x,tspan,
                       args...;
                       kwargs...)
    error("neural_ode_rd has been deprecated with the change to Zygote. Please see the documentation on the new NeuralODE layer.")
end

struct NeuralODE{P,M,RE,T,S,A,K}
    p::P
    model::M
    re::RE
    tspan::T
    solver::S
    args::A
    kwargs::K
end

function NeuralODE(model,tspan,solver=nothing,args...;kwargs...)
    p,re = Flux.destructure(model)
    NeuralODE(p,model,re,tspan,solver,args,kwargs)
end

Flux.@functor NeuralODE

function (n::NeuralODE)(x,p=n.p)
    dudt_(u,p,t) = n.re(p)(u)
    prob = ODEProblem{false}(dudt_,x,n.tspan,p)
    concrete_solve(prob,n.solver,x,p,n.args...;n.kwargs...)
end

function neural_dmsde(model,x,mp,tspan,
                      args...;kwargs...)
    error("neural_dmsde has been deprecated with the change to Zygote. Please see the documentation on the new NeuralDSDE layer.")
end

struct NeuralDSDE{P,M,RE,M2,RE2,T,S,A,K}
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
end

function NeuralDSDE(model1,model2,tspan,solver=nothing,args...;kwargs...)
    p1,re1 = Flux.destructure(model1)
    p2,re2 = Flux.destructure(model2)
    p = [p1;p2]
    NeuralDSDE(p,length(p1),model1,re1,model2,re2,tspan,solver,args,kwargs)
end

Flux.@functor NeuralDSDE

function (n::NeuralDSDE)(x,p=n.p)
    dudt_(u,p,t) = n.re1(p[1:n.len])(u)
    g(u,p,t) = n.re2(p[(n.len+1):end])(u)
    prob = SDEProblem{false}(dudt_,g,x,n.tspan,p)
    concrete_solve(prob,n.solver,x,p,n.args...;sensealg=TrackerAdjoint(),n.kwargs...)
end

struct NeuralSDE{P,M,RE,M2,RE2,T,S,A,K}
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

struct NeuralCDDE{P,M,RE,H,L,T,S,A,K}
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
