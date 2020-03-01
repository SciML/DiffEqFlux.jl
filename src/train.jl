struct NullData end
const DEFAULT_DATA = Iterators.cycle((NullData(),))
Base.iterate(::NullData, i=1) = nothing
Base.length(::NullData) = 0

get_maxiters(data) = Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.IsInfinite ||
                     Iterators.IteratorSize(typeof(DEFAULT_DATA)) isa Iterators.SizeUnknown ?
                     typemax(Int) : length(data)

function update!(x::AbstractArray, x̄::AbstractArray{<:ForwardDiff.Dual})
  x .-= x̄
end

function update!(x::AbstractArray, x̄)
  x .-= getindex.(ForwardDiff.partials.(x̄),1)
end

function update!(opt, x, x̄)
  x .-= Flux.Optimise.apply!(opt, x, x̄)
end

function update!(opt, x, x̄::AbstractArray{<:ForwardDiff.Dual})
  x .-= Flux.Optimise.apply!(opt, x, getindex.(ForwardDiff.partials.(x̄),1))
end

function update!(opt, xs::Flux.Zygote.Params, gs)
  for x in xs
    gs[x] == nothing && continue
    update!(opt, x, gs[x])
  end
end

"""
    sciml_train(loss, θ, opt, data = DEFAULT_DATA; cb, maxiters)

Optimizes the `loss(θ,curdata...)` function with respect to the parameter vector
`θ` iterating over the `data`. By default the data iterator is empty, i.e.
`loss(θ)` is used. The first output of the loss function is considered the loss.
Extra outputs are passed to the callback.

The keyword arguments are as follows:

- `cb`: For specifying a callback function `cb(θ,l...)` which acts on the current
  parameters `θ`, where `l...` are the returns of the latest call to `loss`. This
  callback should return a boolean where, if true, the training will end prematurely.
- `maxiters`: Specifies the maximum number of iterations for the optimization.
  Required if a Flux optimizer is chosen and no data iterator is given, otherwise
  defaults to infinite.
"""
function sciml_train(loss, _θ, opt, _data = DEFAULT_DATA;
                     cb = (args...) -> (false),
                     maxiters = get_maxiters(data))

  # Flux is silly and doesn't have an abstract type on its optimizers, so assume
  # this is a Flux optimizer
  θ = copy(_θ)
  ps = Flux.params(θ)

  if _data == DEFAULT_DATA && maxiters == typemax(Int)
    error("For Flux optimizers, either a data iterator must be provided or the `maxiters` keyword argument must be set.")
  elseif _data == DEFAULT_DATA && maxiters != typemax(Int)
    data = Iterators.repeated((), maxiters)
  elseif maxiters != typemax(Int)
    data = take(_data, maxiters)
  else
    data = _data
  end

  t0 = time()
  local x
  @progress for d in data
    gs = Flux.Zygote.gradient(ps) do
      x = loss(θ,d...)
      first(x)
    end
    cb_call = cb(θ,x...)
    if !(typeof(cb_call) <: Bool)
      error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
    elseif cb_call
      break
    end
    
    update!(opt, ps, gs)
  end

  _time = time()
  Optim.MultivariateOptimizationResults(opt,
                                        _θ,# initial_x,
                                        θ, #pick_best_x(f_incr_pick, state),
                                        first(x), # pick_best_f(f_incr_pick, state, d),
                                        maxiters, #iteration,
                                        maxiters >= maxiters, #iteration == options.iterations,
                                        false, # x_converged,
                                        0.0,#T(options.x_tol),
                                        0.0,#T(options.x_tol),
                                        NaN,# x_abschange(state),
                                        NaN,# x_abschange(state),
                                        false,# f_converged,
                                        0.0,#T(options.f_tol),
                                        0.0,#T(options.f_tol),
                                        NaN,#f_abschange(d, state),
                                        NaN,#f_abschange(d, state),
                                        false,#g_converged,
                                        0.0,#T(options.g_tol),
                                        NaN,#g_residual(d),
                                        false, #f_increased,
                                        nothing,
                                        maxiters,
                                        maxiters,
                                        0,
                                        true,
                                        NaN,
                                        _time-t0,)
end

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function sciml_train(loss, θ, opt::Optim.AbstractOptimizer, data = DEFAULT_DATA;
                      cb = (args...) -> (false), maxiters = get_maxiters(data))
  local x, cur, state
  cur,state = iterate(data)

  function _cb(trace)
    cb_call = cb(decompose_trace(trace).metadata["x"],x...)
    if !(typeof(cb_call) <: Bool)
      error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
    end
    cur,state = iterate(data,state)
    cb_call
  end

  function optim_fg!(F,G,θ)
    _x,lambda = Flux.Zygote.pullback(θ) do θ
      x = loss(θ,cur...)
      first(x)
    end

    if G != nothing
      grad = first(lambda(1))
      if eltype(grad) <: ForwardDiff.Dual
        G .= getindex.(ForwardDiff.partials.(grad),1)
      else
        G .= grad
      end
    end

    if F != nothing
      return _x
    end

    return _x
  end

  optimize(Optim.only_fg!(optim_fg!), θ, opt,
           Optim.Options(extended_trace=true,callback = _cb,
                         f_calls_limit = maxiters))
end

function sciml_train(loss, θ, opt::Optim.AbstractConstrainedOptimizer,
                     data = DEFAULT_DATA;
                     lower_bounds, upper_bounds,
                     cb = (args...) -> (false), maxiters = get_maxiters(data))
  local x, cur, state
  cur,state = iterate(data)

  function _cb(trace)
    cb_call = cb(decompose_trace(trace).metadata["x"],x...)
    if !(typeof(cb_call) <: Bool)
      error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
    end
    cur,state = iterate(data,state)
    cb_call
  end

  function optim_fg!(F,G,θ)
    _x,lambda = Flux.Zygote.pullback(θ) do θ
      x = loss(θ,cur...)
      first(x)
    end

    if G != nothing
      G .= first(lambda(1))
    end

    if F != nothing
      return _x
    end

    return _x
  end

  optimize(Optim.only_fg!(optim_fg!), lower_bounds, upper_bounds, θ, opt,
           Optim.Options(extended_trace=true,callback = _cb,
                         f_calls_limit = maxiters))
end
