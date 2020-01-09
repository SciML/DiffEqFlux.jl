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

# Flux Layer Interface
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

# Play nice with Flux
Flux.@functor NeuralODE

function (n::NeuralODE)(x)
    dudt_(u,p,t) = n.re(p)(u)
    prob = ODEProblem{false}(dudt_,x,n.tspan,n.p)
    concrete_solve(prob,n.solver,x,n.p,n.args...;n.kwargs...)
end

function neural_dmsde(model,x,mp,tspan,
                      args...;kwargs...)
    error("neural_dmsde has been deprecated with the change to Zygote. Please see the documentation on the new NeuralDMSDE layer.")
end

# Flux Layer Interface
struct NeuralDMSDE{P,M,RE,MP,T,S,A,K}
    p::P
    model::M
    re::RE
    mp::MP
    tspan::T
    solver::S
    args::A
    kwargs::K
end

function NeuralDMSDE(model,mp,tspan,solver=nothing,args...;kwargs...)
    p,re = Flux.destructure(model)
    NeuralDMSDE(p,model,re,mp,tspan,solver,args,kwargs)
end

# Play nice with Flux
Flux.@functor NeuralDMSDE

function (n::NeuralDMSDE)(x)
    dudt_(u,p,t) = n.re(n.p)(u)
    g(u,p,t) = n.mp.*u
    prob = SDEProblem{false}(dudt_,g,x,n.tspan,n.p)
    #diffeq_rd(n.p,prob,n.solver,n.args...;u0=x,n.kwargs...)
    concrete_solve(prob,n.solver,x,n.p,n.args...;sensealg=TrackerAdjoint(),n.kwargs...)
end
