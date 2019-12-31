## Reverse-Mode via Tracker.jl

diffeq_rd(p,prob,args...;u0=prob.u0,kwargs...) = _diffeq_rd(p,prob,u0,args...;kwargs...)

function _diffeq_rd(p,prob,u0,args...;kwargs...)
  if typeof(u0) <: AbstractArray && !(typeof(u0) <: TrackedArray)
    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=convert.(recursive_bottom_eltype(p),u0),p=p)
    else
      # use TrackedArray for efficiency of the tape
      _prob = remake(prob,u0=u0,p=p)
    end
  else # u0 is functional, ignore the change
    _prob = remake(prob,u0=u0,p=p)
  end
  Array(solve(_prob,args...;kwargs...))
end

ZygoteRules.@adjoint function _diffeq_rd(p,prob,u0,args...;kwargs...)
  function tracker_forward(u0,p)
    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=map(identity,u0),p=p)
    else
      # use TrackedArray for efficiency of the tape
      _prob = remake(prob,u0=u0,p=p)
    end
    Array(solve(_prob,args...;kwargs...))
  end
  val,pullback = Tracker.forward(tracker_forward,u0,p)
  Tracker.data(val), function (ybar)
    u0bar,pbar = pullback(ybar)
    _u0bar = u0bar isa Tracker.TrackedArray ? Tracker.data(u0bar) : Tracker.data.(u0bar)
    (Tracker.data(pbar),nothing,_u0bar,ntuple(_->nothing, length(args))...)
  end
end

ZygoteRules.@adjoint function _diffeq_rd(p::Flux.Params,prob,u0,args...;kwargs...)
  function tracker_forward(u0,p)
    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=map(identity,u0),p=p)
    else
      # use TrackedArray for efficiency of the tape
      _prob = remake(prob,u0=u0,p=p)
    end
    Array(solve(_prob,args...;kwargs...))
  end
  non_u0_p = p.order[p.order .!== (u0,)]
  _p = reduce(vcat,vec.(non_u0_p))
  sz = size.(non_u0_p)
  val,pullback = Tracker.forward(tracker_forward,u0,_p)
  val, function (ybar)
    u0bar, pbar = pullback(ybar)
    _pbar = Tracker.data(pbar)
    rs = restructure(tuple(sz...), _pbar)
    graddict = IdDict(x=>y for (x,y) in zip(non_u0_p,rs))
    for x in keys(__context__.cache)
      if x === u0
        __context__.cache[x] = Tracker.data(u0bar)
      else
        __context__.cache[x] = graddict[x]
      end
    end
    (_pbar,nothing,Tracker.data(u0bar),ntuple(_->nothing, length(args))...)
  end
end

## Forward-Mode via ForwardDiff.jl

function diffeq_fd(p,f,n,prob,args...;u0=prob.u0,kwargs...)
  _prob = remake(prob,u0=convert.(eltype(p),u0),p=p)
  f(solve(_prob,args...;kwargs...))
end

ZygoteRules.@adjoint function diffeq_fd(p,f,n,prob,args...;u0=prob.u0,kwargs...)
  function diffeq_fd_forward(p)
    _prob = remake(prob,u0=convert.(eltype(p),u0),p=p)
    f(solve(_prob,args...;kwargs...))
  end
  _p = Tracker.data(p)
  if n === nothing
    result = DiffResults.GradientResult(_p)
    ForwardDiff.gradient!(result, diffeq_fd_forward, _p)
    function diffeq_fd_adjoint1(Δ)
      (Δ .* DiffResults.gradient(result), ntuple(_->nothing, 3+length(args))...)
    end
    DiffResults.value(result),diffeq_fd_adjoint1
  else
    y = adapt(typeof(_p),zeros(n))
    result = DiffResults.JacobianResult(y,_p)
    ForwardDiff.jacobian!(result, diffeq_fd_forward, _p)
    function diffeq_fd_adjoint2(Δ)
      (DiffResults.jacobian(result)' * Δ, ntuple(_->nothing, 3+length(args))...)
    end
    DiffResults.value(result),diffeq_fd_adjoint2
  end
end

## Reverse-Mode using Adjoint Sensitivity Analysis
# Always reduces to Array

diffeq_adjoint(p,prob,args...;u0=prob.u0,kwargs...) = _diffeq_adjoint(p,u0,prob,args...;kwargs...)

function _diffeq_adjoint(p,u0,prob,args...;kwargs...)
  _prob = remake(prob,u0=u0,p=p)
  T = gpu_or_cpu(u0)
  adapt(T, solve(_prob,args...;kwargs...))
end

ZygoteRules.@adjoint function _diffeq_adjoint(p,u0,prob,args...;
                                              save_start=true,save_end=true,
                                              kwargs...)

  T = gpu_or_cpu(u0)
  _prob = remake(prob,u0=u0,p=p)

  # Force `save_start` and `save_end` in the forward pass This forces the
  # solver to do the backsolve all the way back to `u0` Since the start aliases
  # `_prob.u0`, this doesn't actually use more memory But it cleans up the
  # implementation and makes `save_start` and `save_end` arg safe.
  kwargs_fwd = NamedTuple{Base.diff_names(Base._nt_names(
  values(kwargs)), (:callback_adj,))}(values(kwargs))
  kwargs_adj = NamedTuple{Base.diff_names(Base._nt_names(values(kwargs)), (:callback_adj,:callback))}(values(kwargs))
  if haskey(kwargs, :callback_adj)
    kwargs_adj = merge(kwargs_adj, NamedTuple{(:callback,)}( [get(kwargs, :callback_adj, nothing)] ))
  end
  sol = solve(_prob,args...;save_start=true,save_end=true,kwargs...)

  no_start = !save_start
  no_end = !save_end
  sol_idxs = 1:length(sol)
  no_start && (sol_idxs = sol_idxs[2:end])
  no_end && (sol_idxs = sol_idxs[1:end-1])
  # If didn't save start, take off first. If only wanted the end, return vector
  only_end = length(sol_idxs) <= 1
  u = sol[sol_idxs]
  only_end && (sol_idxs = length(sol))
  out = only_end ? sol[end] : reduce((x,y)->cat(x,y,dims=ndims(u)),u.u)

  function diffeq_adjoint_adjoint(Δ)
    function df(out, u, p, t, i)
      if only_end
        out[:] .= -vec(Δ)
      else
        out[:] .= -reshape(Δ, :, size(Δ)[end])[:, i]
      end
    end

    ts = sol.t[sol_idxs]
    du0, dp = adjoint_sensitivities_u0(sol,args...,df,ts;kwargs_adj...)
    if p isa Flux.Params
      rs = restructure(tuple(size.(p)...), dp')
      graddict = IdDict(x=>y for (x,y) in zip(p.order,rs))
      for x in keys(__context__.cache)
        __context__.cache[x] = graddict[x]
      end
    else
      rs = reshape(dp', size(p))
    end
    (rs, reshape(du0,size(u0)), ntuple(_->nothing, 1+length(args))...)
  end
  out,diffeq_adjoint_adjoint
end
