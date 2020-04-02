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

abstract type DiffMode end
struct ForwardDiffMode <: DiffMode end
struct TrackerDiffMode <: DiffMode end
struct ReverseDiffMode <: DiffMode end
struct ZygoteDiffMode <: DiffMode end

macro withprogress(progress, exprs...)
  quote
    if $progress
      $DiffEqBase.maybe_with_logger($DiffEqBase.default_logger($Logging.current_logger())) do
        $ProgressLogging.@withprogress $(exprs...)
      end
    else
      $(exprs[end])
    end
  end |> esc
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
- `save_best`: Specifies whether you want the best solution, corresponding to the
  lowest loss function value. If false, the last solution found will be returned.
"""
function sciml_train(loss, _θ, opt, _data = DEFAULT_DATA;
                     cb = (args...) -> false,
                     maxiters = get_maxiters(data),
                     progress=true, save_best=true)

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

  local x, min_err
  min_err = typemax(eltype(_θ)) #dummy variables
  min_opt = 1
  
  @withprogress progress name="Training" begin
    for (i,d) in enumerate(data)
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
      msg = @sprintf("loss: %.3g", x[1])
      progress && ProgressLogging.@logprogress msg i/maxiters
      update!(opt, ps, gs)

      if save_best
        if first(x) < first(min_err)  #found a better solution
          min_opt = opt
          min_err = x
        end
        if i == maxiters  #Last iteration, revert to best.
          opt = min_opt
          cb(θ,min_err...)
        end
      end
    end
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
                                        _time-t0)
end

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function sciml_train(loss, θ, opt::Optim.AbstractOptimizer, data = DEFAULT_DATA;
                      cb = (args...) -> (false), maxiters = get_maxiters(data),
                      diffmode = ZygoteDiffMode(), allow_f_increases=true, kwargs...)
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

  _loss = function (θ)
    x = loss(θ,cur...)
    first(x)
  end

  if diffmode isa ZygoteDiffMode
    f!  =  function (θ)
      _loss(θ)
    end

    g!  =  function (G,θ)
      _x,lambda = Flux.Zygote.pullback(_loss,θ)
      grad = first(lambda(1))
      if eltype(grad) <: ForwardDiff.Dual
        G .= getindex.(ForwardDiff.partials.(grad),1)
      else
        G .= grad
      end
      return _x
    end

    fg! =  function (G,θ)
      _x,lambda = Flux.Zygote.pullback(_loss,θ)
      if G != nothing
        grad = first(lambda(1))
        if eltype(grad) <: ForwardDiff.Dual
          G .= getindex.(ForwardDiff.partials.(grad),1)
        else
          G .= grad
        end
      end
      return _x
    end

    if opt isa Optim.KrylovTrustRegion
      hv! = function (H,θ,v)
        _θ = ForwardDiff.Dual.(θ,v)
        H .= getindex.(ForwardDiff.partials.(Flux.Zygote.gradient(_loss,_θ)[1]),1)
      end
      optim_f = Optim.TwiceDifferentiableHV(f!,fg!,hv!,θ)
    else
      h! = function (H,θ)
        H .= ForwardDiff.jacobian(θ) do θ
          Flux.Zygote.gradient(_loss,θ)[1]
        end
      end
      optim_f = TwiceDifferentiable(f!, g!, fg!, h!, θ)
    end

  else
    if diffmode isa TrackerDiffMode
      optim_fg! =  function (F,G,θ)
        _x,lambda = Tracker.forward(_loss,θ)
        if G != nothing
          G .= first(lambda(1))
        end
        if F != nothing
          return _x
        end
        return _x
      end
    elseif diffmode isa ForwardDiffMode
      optim_fg! = let res = DiffResults.GradientResult(θ)
        function (F,G,θ)
          if G != nothing
            ForwardDiff.gradient!(res,_loss,θ)
            G .= DiffResults.gradient(res)
          end

          if F != nothing
            return DiffResults.value(res)
          end

          return _x
        end
      end
    elseif diffmode isa ReverseDiffMode
      optim_fg! = let res = DiffResults.GradientResult(θ)
        function (F,G,θ)
          if G != nothing
            ReverseDiff.gradient!(res,_loss,θ)
            G .= DiffResults.gradient(res)
          end

          if F != nothing
            return DiffResults.value(res)
          end

          return _x
        end
      end
    end
    optim_f = Optim.only_fg!(optim_fg!)
  end
  Optim.optimize(optim_f, θ, opt,
           Optim.Options(;extended_trace=true,callback = _cb,
                         f_calls_limit = maxiters,
                         allow_f_increases=allow_f_increases,
                         kwargs...))
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

  Optim.optimize(Optim.only_fg!(optim_fg!), lower_bounds, upper_bounds, θ, opt,
           Optim.Options(extended_trace=true,callback = _cb,
                         f_calls_limit = maxiters))
end

struct BBO
  method::Symbol         
end

BBO() = BBO(:adaptive_de_rand_1_bin)

function sciml_train(loss, opt::BBO = BBO(), data = DEFAULT_DATA;lower_bounds, upper_bounds,
                      maxiters = get_maxiters(data), kwargs...)
  local x, cur, state
  cur,state = iterate(data)

  _loss = function (θ)
    x = loss(θ,cur...)
    first(x)
  end

  bboptre = BlackBoxOptim.bboptimize(_loss;Method = opt.method, SearchRange = [(lower_bounds[i], upper_bounds[i]) for i in 1:length(lower_bounds)], MaxSteps = maxiters, kwargs...)

  Optim.MultivariateOptimizationResults(opt.method,
                                        [NaN],# initial_x,
                                        best_candidate(bboptre), #pick_best_x(f_incr_pick, state),
                                        best_fitness(bboptre), # pick_best_f(f_incr_pick, state, d),
                                        bboptre.iterations, #iteration,
                                        bboptre.iterations >= maxiters, #iteration == options.iterations,
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
                                        bboptre.elapsed_time)
end
