function diffeq_fd(p,f,prob,args...;kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  f(solve(_prob,args...;kwargs...))
end
Flux.Tracker.@grad function diffeq_fd(p::TrackedVector,f,prob,args...;kwargs...)
  _f = function (p)
    _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
    f(solve(_prob,args...;kwargs...))
  end
  result = DiffResults.GradientResult(p)
  ForwardDiff.gradient!(result, _f, p)
  DiffResults.value(res),Δ -> (Δ .* DiffResults.gradient(res))
end

function diffeq_adjoint(p::TrackedVector,prob,args...;kwargs...)
  solve(remake(prob,u0=eltype(p).(prob.u0),p=p),args...;kwargs...)
end
# Example: https://github.com/JuliaDiffEq/DiffEqSensitivity.jl/blob/master/test/adjoint.jl
# How to provide the cost function here? Is this fine?
Flux.Tracker.@grad function diffeq_adjoint(prob,p,dg,t,args...;kwargs...)
  sol = solve(remake(prob,u0=eltype(p).(prob.u0),p=p),args...;kwargs...)
  adjoint_sensitivities(sol,args...,dg,t,kwargs...)
end

function diffeq_rd(p::TrackedVector,prob,args...;kwargs...)
  solve(remake(prob,u0=eltype(p).(prob.u0),p=p),args...;kwargs...)
end
# No `@grad` since it will automatically use Flux reverse-mode.
