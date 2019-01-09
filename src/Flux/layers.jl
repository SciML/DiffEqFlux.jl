## Reverse-Mode via Flux.jl

function diffeq_rd(p,f,prob,args...;kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  f(solve(_prob,args...;kwargs...))
end

## Forward-Mode via ForwardDiff.jl

function diffeq_fd(p,f,n,prob,args...;kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  f(solve(_prob,args...;kwargs...))
end
diffeq_fd(p::TrackedVector,args...;kwargs...) = Flux.Tracker.track(diffeq_fd, p, args...; kwargs...)
Flux.Tracker.@grad function diffeq_fd(p::TrackedVector,f,n,prob,args...;kwargs...)
  _f = function (p)
    _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
    f(solve(_prob,args...;kwargs...))
  end
  _p = Flux.data(p)
  if n === nothing
    result = DiffResults.GradientResult(_p)
    ForwardDiff.gradient!(result, _f, _p)
    DiffResults.value(result),Δ -> (Δ .* DiffResults.gradient(result),nothing,nothing,nothing,map(_ -> nothing, args)...)
  else
    y = zeros(n)
    result = DiffResults.JacobianResult(y,_p)
    ForwardDiff.jacobian!(result, _f, _p)
    DiffResults.value(result),Δ -> (DiffResults.jacobian(result)' * Δ,nothing,nothing,nothing,map(_ -> nothing, args)...)
  end
end

## Reverse-Mode using Adjoint Sensitivity Analysis

function diffeq_adjoint(p,prob,f,df,t,args...;kwargs...)
  _prob = remake(prob,p=p)
  f(solve(_prob,args...;kwargs...))
end

diffeq_adjoint(p::TrackedVector,args...;kwargs...) = Flux.Tracker.track(diffeq_adjoint, p, args...; kwargs...)
# Example: https://github.com/JuliaDiffEq/DiffEqSensitivity.jl/blob/master/test/adjoint.jl
# How to provide the cost function here? Is this fine?
Flux.Tracker.@grad function diffeq_adjoint(p::TrackedVector,f,df,t,prob,args...;kwargs...)
  _prob = remake(prob,p=p.data)
  sol = solve(_prob,args...;kwargs...)
  f(sol), Δ -> begin
    grad = adjoint_sensitivities(sol,args...,df,t;sensealg=SensitivityAlg(autojacvec=false),kwargs...)
    (Δ .* grad', ntuple(_->nothing, 4+length(args))...)
  end
end
