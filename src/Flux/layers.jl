function diffeq_fd(p,f,prob,args...;kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  f(solve(_prob,args...;kwargs...))
end
function diffeq_adjoint(p,prob,args...;kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  f(solve(_prob,args...;kwargs...))
end
function diffeq_rd(p,f,prob,args...;kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  f(solve(_prob,args...;kwargs...))
end

Flux.Tracker.@grad function diffeq_fd(p::TrackedVector,f,prob,args...;kwargs...)
  _f = function (p)
    _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
    f(solve(_prob,args...;kwargs...))
  end
  _p = Flux.data(p)
  result = DiffResults.GradientResult(_p)
  ForwardDiff.gradient!(result, _f, _p)
  DiffResults.value(res),Δ -> (Δ .* DiffResults.gradient(res))
end

# Example: https://github.com/JuliaDiffEq/DiffEqSensitivity.jl/blob/master/test/adjoint.jl
# How to provide the cost function here? Is this fine?
Flux.Tracker.@grad function diffeq_adjoint(prob,p,dg,t,args...;kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  sol = solve(_prob,args...;kwargs...)
  f(sol),Δ -> (Δ .* adjoint_sensitivities(sol,args...,dg,t,kwargs...))
end
