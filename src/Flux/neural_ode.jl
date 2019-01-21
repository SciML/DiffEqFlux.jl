neural_ode_reduction(sol) = Array(sol)
function neural_ode(x,model,tspan,ad_func,args...;kwargs...)
  p = Flux.data(destructure(model))
  dudt_(du,u::TrackedArray,p,t) = du .= restructure(model,p)(u)
  dudt_(du,u::AbstractArray,p,t) = du .= Flux.data(restructure(model,p)(u))
  prob = ODEProblem(dudt_,x,tspan,p)

  if ad_func === diffeq_adjoint
    return ad_func(p,prob,args...;kwargs...)
  elseif ad_func === diffeq_fd
    return ad_func(p,neural_ode_reduction,length(p),prob,args...;kwargs...)
  else
    return ad_func(p,neural_ode_reduction,prob,args...;kwargs...)
  end
end
