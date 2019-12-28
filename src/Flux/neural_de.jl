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
function neural_ode(model,x,tspan,args...;p=Flux.params(x,model),kwargs...)
  dudt_(u::AbstractArray,p,t) = model(u)
  prob = ODEProblem{false}(dudt_,x,tspan,p)
  return diffeq_adjoint(p,prob,args...;u0=x,kwargs...)
end


"""
Constructs a neural ODE with the gradients computed using  reverse-mode
automatic differentiation. This is equivalent to discretizing then optimizing
the differential equation, cf neural_ode for a comparison with the adjoint method.
"""
function neural_ode_rd(model,x,tspan,
                       args...;kwargs...)
  dudt_(u,p,t) = model(u)
  prob = ODEProblem{false}(dudt_,x,tspan)
  p = destructure(model)
  Array(diffeq_rd(p,prob,args...;u0=x,kwargs...))
end

function neural_dmsde(model,x,mp,tspan,
                      args...;kwargs...)
  dudt_(u,p,t) = model(u)
  g(u,p,t) = mp.*u
  prob = SDEProblem{false}(dudt_,g,x,tspan,nothing)
  p = destructure(model)
  Array(diffeq_rd(p,prob,args...;u0=x,kwargs...))
end
