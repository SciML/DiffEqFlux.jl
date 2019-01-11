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
    DiffResults.value(result),Δ -> (Δ .* DiffResults.gradient(result), ntuple(_->nothing, 3+length(args))...)
  else
    y = zeros(n)
    result = DiffResults.JacobianResult(y,_p)
    ForwardDiff.jacobian!(result, _f, _p)
    DiffResults.value(result),Δ -> (DiffResults.jacobian(result)' * Δ, ntuple(_->nothing, 3+length(args))...)
  end
end

## Reverse-Mode using Adjoint Sensitivity Analysis

function diffeq_adjoint(p,prob,args...;kwargs...)
  _prob = remake(prob,p=p.data)
  Array(solve(_prob,args...;kwargs...))
end

diffeq_adjoint(p::TrackedVector,args...;kwargs...) = Flux.Tracker.track(diffeq_adjoint, p, args...; kwargs...)
# Example: https://github.com/JuliaDiffEq/DiffEqSensitivity.jl/blob/master/test/adjoint.jl
# How to provide the cost function here? Is this fine?
Flux.Tracker.@grad function diffeq_adjoint(p::TrackedVector,prob,args...;kwargs...)
  _prob = remake(prob,p=Flux.data(p))
  sol = solve(_prob,args...;kwargs...)
  Array(sol), Δ -> begin
    Δ = Flux.data(Δ)
    ts = sol.t
    df(out, u, p, t, i) = @. out = - @view Δ[:, i]
    grad = adjoint_sensitivities(sol,args...,df,ts;kwargs...)
    (grad', ntuple(_->nothing, 4+length(args))...)
  end
end

function diffeq_backwardsolve_adjoint(p, prob, ulen, args...;kwargs...)
  params = p.data
  u = @view params[1:ulen]
  p = @view params[ulen+1:end-2]
  t0, t1 = parms[end-1], parms[end]
  _prob = remake(prob, u0=u, p=p, tspan=(t0, t1))
  Array(solve(_prob,args...;kwargs...))
end

Flux.Tracker.@grad function diffeq_backwardsolve_adjoint(p::TrackedVector, prob, ulen, args...; kwargs...)
  params = p.data
  u = @view params[1:ulen]
  p = @view params[ulen+1:end-2]
  t0, t1 = parms[end-1], parms[end]
  _prob = remake(prob, u0=u, p=p, tspan=(t0, t1))
  sol = solve(_prob,args...;kwargs...)
  Array(sol), Δ -> begin
    tspan=(t1, t0)
    p = sol.prob.p
    ts = sol.t
    f = sol.prob.f
    ii = Ref(size(Δ, 2))
    time_choice(integrator) = ii[] > 0 ? ts[ii[]] : nothing
    function affect!(integrator)
      p, z = integrator.p, integrator.u
      uidx = 1:ulen
      plen = (length(z)-ulen)÷2
      λidx = ulen+1:ulen+1+pen
      #intλidx = ulen+1+pen:length(z)
      #intλ = @view z[intλidx]
      #@. intλ -= Δ[uidx+ulen, ii[]]
      λ = @view z[λidx]
      @. λ -= @view Δ[uidx, ii[]]
      u_modified!(integrator,true)
      ii[] -= 1
      return nothing
    end
    function aug_dynamics(dz, z, p, t) # z is [u; z; sens]
      uidx = 1:ulen
      plen = (length(z)-ulen)÷2
      λidx = ulen+1:ulen+1+pen
      @views du, u = dz[uidx], z[uidx]
      f(du, u, p, t)
      # vjp ???
      return nothing
    end
    dLdt = @view(Δ[:, end])'f(...)
    state0 = [sol[end]; f(); zeros(length(params) - ulen); ...]
    cb = IterativeCallback(time_choice,affect!,eltype(tspan);initial_affect=true)
    __prob = remake(prob, u0=u, p=p, tspan=(t0, t1))
    sol = solve(__prob, args...;callback=cb, kwargs...)
  end
end
