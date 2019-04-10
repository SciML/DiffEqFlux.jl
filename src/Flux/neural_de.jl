function neural_ode(model,x,tspan,
                    args...;kwargs...)
  Tracker.istracked(x) && error("u0 is not currently differentiable.")
  p = destructure(model)
  dudt_(du,u::TrackedArray,p,t) = du .= restructure(model,p)(u)
  dudt_(du,u::AbstractArray,p,t) = du .= Flux.data(restructure(model,p)(u))
  prob = ODEProblem(dudt_,x,tspan,p)
  return diffeq_adjoint(p,prob,args...;kwargs...)
end

function neural_ode_rd(model,x,tspan,
                    args...;kwargs...)
  Tracker.istracked(x) && error("u0 is not currently differentiable.")
  p = destructure(model)
  dudt_(u::TrackedArray,p,t) = restructure(model,p)(u)
  dudt_(u::AbstractArray,p,t) = Flux.data(restructure(model,p)(u))
  prob = ODEProblem(dudt_,x,tspan,p)
  return Flux.Tracker.collect(diffeq_rd(p,prob,args...;kwargs...))
end

neural_msde(x,model,mp,tspan,args...;kwargs...) = neural_msde(x,model,mp,tspan,
                                                         diffeq_fd,
                                                         args...;kwargs...)
function neural_dmsde(model,x,mp,tspan,
                      args...;kwargs...)
  Tracker.istracked(x) && error("u0 is not currently differentiable.")
  p = destructure(model)
  dudt_(u::TrackedArray,p,t) = restructure(model,p)(u)
  dudt_(u::AbstractArray,p,t) = Flux.data(restructure(model,p)(u))
  g(u,p,t) = mp.*u
  prob = SDEProblem(dudt_,g,x,tspan,p)
  return Flux.Tracker.collect(diffeq_rd(p,prob,args...;kwargs...))
end
