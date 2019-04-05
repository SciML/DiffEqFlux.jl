function neural_ode(model,x,tspan,
                    args...;kwargs...)
  p = destructure(model)
  dudt_(du,u::TrackedArray,p,t) = du .= restructure(model,p)(u)
  dudt_(du,u::AbstractArray,p,t) = du .= Flux.data(restructure(model,p)(u))
  prob = ODEProblem(dudt_,x,tspan,p)
  return diffeq_adjoint(p,prob,args...;kwargs...)
end

function neural_ode_rd(model,x,tspan,
                       args...;kwargs...)
  dudt_(u,p,t) = model(u)
  prob = ODEProblem(dudt_,param(x),tspan)
  # TODO could probably use vcat rather than collect here
  solve(prob, args...; kwargs...) |> Tracker.collect
end

neural_msde(x,model,mp,tspan,args...;kwargs...) = neural_msde(x,model,mp,tspan,
                                                         diffeq_fd,
                                                         args...;kwargs...)
function neural_dmsde(model,x,mp,tspan,
                      args...;kwargs...)
  dudt_(u,p,t) = model(u)
  g(u,p,t) = mp.*u
  prob = SDEProblem(dudt_,g,param(x),tspan,nothing)
  # TODO could probably use vcat rather than collect here
  solve(prob, args...; kwargs...) |> Tracker.collect
end
